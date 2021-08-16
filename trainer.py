# coding: utf-8
import os
import time
import random
from typing import Dict, Tuple
from pathlib import Path
from logging import Logger

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from tensorboardX import SummaryWriter
try:
    from apex import amp
except ImportError:
    raise ImportError(
        'Please install apex from https://www.github.com/nvidia/apex to use fp16 training.'
    )

from .factories import (
                        loss_fn_factory,
                        optimizer_factory,
                        scheduler_factory,
                        scheduler_with_warmup_factory,
                        get_parameter_names
                        )
from toolFunction.trainer.yaml_config import CfgNode


class Trainer:
    def __init__(self, args: CfgNode, logger: Logger) -> None:
        self.args = args
        self.logger = logger
        self.model: nn.Module = None
        self.loss_fn: nn.Module = None
        self.best_eval_results = {}

    def init_model(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_train_dataloader(self):
        raise NotImplementedError

    def get_eval_dataloader(self):
        raise NotImplementedError

    def get_test_dataloader(self):
        raise NotImplementedError

    def shuffle_data_between_epoch(self):
        raise NotImplementedError

    def create_optimizer(self, forbidden_layer):
        self.logger.info('Creating optimizer')
        decay_parameters = get_parameter_names(self.model, forbidden_layer)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.optimizer.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
        self.optimizer = optimizer_factory(self.args, self.args.optimizer.type, optimizer_grouped_parameters)

    def create_scheduler(
                        self,
                        num_warmup_steps: int,
                        num_training_steps: int):
        self.logger.info('Creating scheduler')
        if self.args.scheduler.warmup:
            self.scheduler = scheduler_with_warmup_factory(
                                                    self.args,
                                                    optimizer=self.optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps,
                                                    last_epoch=self.args.scheduler.get('last_epoch', -1),
                                                    num_cycles=self.args.scheduler.get('num_cycles', 0.5),
                                                    lr_end=self.args.scheduler.get('lr_end', 1e-7),
                                                    power=self.args.scheduler.get('power', 1.0))
        else:
            self.scheduler = scheduler_factory(self.args, optimizer=self.optimizer)

    def create_loss_fn(self):
        self.logger.info('Creating loss function')
        self.loss_fn = loss_fn_factory(self.args)

    def compute_loss(self, outputs, labels) -> torch.Tensor:
        return self.loss_fn(outputs, labels)

    def get_inputs(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoded, labels = batch
        inputs = {
                    'input_ids': torch.stack([item['input_ids'] for item in encoded]).squeeze(1).cuda(self.args.model.device),
                    'token_type_ids': torch.stack([item['token_type_ids'] for item in encoded]).squeeze(1).cuda(self.args.model.device),
                    'attention_mask': torch.stack([item['attention_mask'] for item in encoded]).squeeze(1).cuda(self.args.model.device),
                    'vec_type': self.args.common.vec_type,
                    'pool': self.args.common.pool
                 }
        return inputs, labels

    def set_best_save_path(self):
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        tail = time.strftime('%m-%d_%H.%M', time.localtime()) + '_{}_lr_{}_seed_{}'.format(self.args.training.model_type, lr, self.args.training.seed) + '_best_model'
        self.args.training.best_save_path = os.path.join(self.args.training.save_path, tail)
        if not os.path.exists(self.args.training.best_save_path):
            os.makedirs(self.args.training.best_save_path)

    def train(self, forbidden_layer=[nn.LayerNorm]):
        ''' Train the model '''
        self.seed_everything()
        self.init_model()
        self.create_optimizer(forbidden_layer)
        self.create_loss_fn()
        if self.args.model.continue_training:
            if not os.path.exists(self.args.model.continue_training_path):
                raise FileNotFoundError('continue_training_path not exists')
            self.logger.info('Loading continue training state_dict...')
            self.from_pretrained(self.args.training.ab_initio)
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()
        test_dataloader = self.get_test_dataloader()

        if self.args.training.max_steps > 0:
            t_total = self.args.training.max_steps
            self.args.training.num_train_epochs = self.args.training.max_steps // (len(train_dataloader) // self.args.training.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.training.gradient_accumulation_steps * self.args.training.num_train_epochs
        self.set_best_save_path()

        # fp16
        if self.args.model.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.args.model.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.model.n_gpu > 1 and self.args.training.do_dp:
            self.logger.info('***** Initialize Data Parallel *****')
            gpus = [int(item) for item in self.args.model.gpus.split()]
            model = nn.DataParallel(self.model, device_ids=gpus, output_device=gpus[0])
            self.args.model.device = gpus[0]
            model.to(self.args.model.device)

        # scheduler
        if self.args.scheduler.use_scheduler:
            num_warmup_steps = int(t_total * self.args.scheduler.warmup_prob)
            self.create_scheduler(num_warmup_steps, t_total)
        # tenorboardx
        log_dir = self.init_log_writer()

        # Train!
        self.logger.info('***** Running training *****')
        self.logger.info('  Num examples = %d', len(train_dataloader))
        self.logger.info('  Num Epochs = %d', self.args.training.num_train_epochs)
        self.logger.info('  Instantaneous batch size = %d', self.args.training.batch_size)
        self.logger.info('  Gradient Accumulation steps = %d', self.args.training.gradient_accumulation_steps)
        self.logger.info('  Total optimization steps = %d', t_total)

        global_step = 0
        self.dev_best_loss = float('inf')
        self.model.zero_grad()

        for epoch in range(int(self.args.training.num_train_epochs)):
            tqdm_iterator = tqdm(train_dataloader)
            for step, batch in enumerate(tqdm_iterator):
                loss_item, outputs, labels = self.training_step(batch)
                if self.args.training.do_adv:
                    self.do_adv(batch)
                if (step + 1) % self.args.training.gradient_accumulation_steps == 0:
                    if self.args.model.fp16:
                        nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), self.args.training.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.training.max_grad_norm)
                    self.optimizer.step()
                    if self.args.scheduler.use_scheduler and self.args.scheduler.step_between_batch:
                        if self.args.scheduler.step_between_epoch:
                            raise TypeError('step_between_epoch should be False when step_between_batch is True')
                        self.step(outputs.cpu(), labels, loss_item)
                    self.model.zero_grad()
                global_step += 1
                if self.args.training.logging_steps > 0 and global_step % self.args.training.logging_steps == 0:
                    lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    self.log(eval_dataloader, loss_item, outputs, labels, global_step, lr)
                desc = 'Training. Epoch: {}/{}, loss: {}'.format(epoch+1, int(self.args.training.num_train_epochs), round(loss_item, 4))
                tqdm_iterator.set_description(desc)
            if self.args.scheduler.use_scheduler and self.args.scheduler.step_between_epoch:
                if self.args.scheduler.step_between_batch:
                    raise TypeError('step_between_batch should be False when step_between_epoch is True')
                self.step(outputs.cpu(), labels, loss_item)
            print('\n')
            if 'cpu' not in str(self.args.model.device):
                torch.cuda.empty_cache()
            if self.args.training.shuffle_data_between_epoch:
                try:
                    train_dataloader = self.shuffle_data_between_epoch()
                except NotImplementedError:
                    self.logger.warn('You are trying to use shuffue_data_between_epoch, but it not implemented, so we skip it')
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log(eval_dataloader, loss_item, outputs, labels, global_step, lr)
        print('\n')
        self.logger.info('***** Best eval results *****')
        info = '-'.join(
                        [f' {key}: {value:.4f} ' for key, value in self.best_eval_results.items()])
        self.logger.info(info)
        if self.args.training.predict:
            self.predict(test_dataloader)
        self.logger.info('***** Finish training *****')
        self.logger.info('TensorBoardX log at {}'.format(log_dir))
        self.logger.info('Best model save at {}'.format(self.args.training.best_save_path))

    def training_step(self, batch):
        inputs, labels = self.get_inputs(batch)
        outputs = self.model(**inputs)
        loss = self.compute_loss(outputs, labels.cuda(self.args.model.device))
        if self.args.model.n_gpu > 1 and self.args.training.do_dp:
            # mean() to average on multi-gpu parallel training
            loss = loss.mean()
        if self.args.training.gradient_accumulation_steps > 1:
            loss = loss / self.args.training.gradient_accumulation_steps
        if self.args.model.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        loss_item = loss.item()
        return loss_item, outputs, labels

    def do_adv(self, batch):
        target = self.args.training.adv_target
        self.model.attack(target)
        inputs, labels = self.get_inputs(batch)
        outputs_adv = self.model(**inputs)
        loss_adv = self.compute_loss(outputs_adv, labels.cuda(self.args.model.device))
        loss_adv.backward()
        self.model.restore(target)

    def evaluate(self, eval_dataloader, data_type: str = 'dev') -> Tuple[Dict[str, float], str]:
        self.model.eval()
        if data_type == 'dev':
            self.logger.info('***** Running evaluation*****')
            self.logger.info('  Num examples = %d', len(eval_dataloader))
            self.logger.info('  Batch size = %d', self.args.training.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        if data_type == 'dev':
            desc = 'Eval'
        else:
            desc = 'Test'
        all_labels = np.array([])
        all_logits = np.array([])
        tqdm_iterator = tqdm(eval_dataloader)
        for batch in tqdm_iterator:
            with torch.no_grad():
                outputs, labels, loss = self.evaluate_step(batch)
                nb_eval_steps += 1
                logits = torch.argmax(outputs, dim=1)
                logits = logits.cpu().numpy()
                all_labels = np.append(all_labels, labels.cpu().numpy())
                all_logits = np.append(all_logits, logits)
                eval_loss += loss
            tqdm_iterator.set_description(desc)

        self.model.train()
        eval_loss = eval_loss / nb_eval_steps
        res = self.metric(all_logits, all_labels, data_type)
        if data_type == 'dev':
            result = res[0]
        else:
            result, report = res
        result['loss'] = eval_loss
        if data_type == 'dev':
            print('\n')
            self.logger.info('***** Eval results *****')
            info = '-'.join([f' {key}: {value:.4f} ' for key, value in result.items()])
            self.logger.info(info)
            return result
        return result, report

    def evaluate_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, float]:
        inputs, labels = self.get_inputs(batch)
        outputs = self.model(**inputs)
        loss = self.compute_loss(outputs, labels.cuda(self.args.model.device))
        if self.args.model.n_gpu > 1 and self.args.training.do_dp:
            #  mean() to average on multi-gpu parallel evaluating
            loss = loss.mean()
        return outputs, labels, loss.item()

    def predict(self, test_dataloader):
        self.load_model()
        results, report = self.evaluate(test_dataloader, data_type='test')
        print('\n')
        self.logger.info('***** Predict results *****')
        info = '-'.join(
            [f' {key}: {value:.4f} ' for key, value in results.items()])
        self.logger.info(info)
        print(report)

    def log(self,
            eval_dataloader,
            loss_item: float,
            outputs: torch.Tensor,
            labels: torch.Tensor,
            global_step: int,
            lr: float):
        logits = torch.argmax(outputs, dim=1)
        logits = logits.cpu().numpy()
        train_report = self.metric(logits, labels.cpu().numpy(), data_type='train')[0]
        print('\n')
        self.logger.info('***** Current train results *****')
        info = '-'.join(
                        [f' {key}: {value:.4f} ' for key, value in train_report.items()])
        self.logger.info(info)
        print('\n')
        eval_report = self.evaluate(eval_dataloader)
        eval_loss = eval_report['loss']
        if eval_loss < self.dev_best_loss:
            self.dev_best_loss = eval_loss
            self.save_pretrained()
            self.best_eval_results = eval_report
        self.writer.add_scalar('loss/train', loss_item, global_step)
        self.writer.add_scalar('loss/eval', eval_loss, global_step)
        self.writer.add_scalar('lr', lr, global_step)
        for key, val in train_report.items():
            title = '{}/train'.format(key)
            self.writer.add_scalar(title, val, global_step)
        for key, val in eval_report.items():
            title = '{}/eval'.format(key)
            self.writer.add_scalar(title, val, global_step)

    def init_tensorboardx(self) -> Path:
        log_dir = os.path.join(
            self.args.training.log_path,
            time.strftime('%m-%d_%H.%M', time.localtime()) + '_{}_seed_{}'.format(self.args.training.model_type, self.args.training.seed))
        self.writer = SummaryWriter(log_dir=log_dir)
        return log_dir

    def save_pretrained(self):
        model_save_path = os.path.join(self.args.training.best_save_path, 'model.pt')
        optimizer_save_path = os.path.join(self.args.training.best_save_path, 'optimizer.pt')
        optimizer_dict = {'name': self.args.optimizer.type, 'state_dict': self.optimizer.state_dict()}
        torch.save(optimizer_dict, optimizer_save_path)
        if self.args.scheduler.use_scheduler:
            scheduler_dict = {'name': self.args.scheduler.type, 'state_dict': self.scheduler.state_dict()}
            scheduler_save_path = os.path.join(self.args.training.best_save_path, 'scheduler.pt')
            torch.save(scheduler_dict, scheduler_save_path)
        self.model.cpu()
        model_to_save = (
                        self.model.module if hasattr(self.model, 'module') else self.model
                        )  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), model_save_path)
        self.logger.info('***** Saving best model to %s *****', model_save_path)
        self.model.to(self.args.model.device)

    def from_pretrained(self, ab_initio: bool):
        model_save_path = os.path.join(self.args.model.continue_training_path, 'model.pt')
        optimizer_save_path = os.path.join(self.args.model.continue_training_path, 'optimizer.pt')
        scheduler_save_path = os.path.join(self.args.model.continue_training_path, 'scheduler.pt')
        self.model.cpu()
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.to(self.args.model.device)
        if not ab_initio:
            if os.path.exists(optimizer_save_path):
                optimizer_dict = torch.load(optimizer_save_path)
                if self.args.optimizer.type == optimizer_dict.get('name'):
                    self.logger.info('***** Loading optimizer state dict *****')
                    self.optimizer.load_state_dict(optimizer_dict.get('state_dict'))
                else:
                    self.logger.info('***** Your optimizer is different from the saved *****')
            if os.path.exists(scheduler_save_path):
                if self.args.scheduler.use_scheduler:
                    scheduler_dict = torch.load(scheduler_save_path)
                    if self.args.scheduler.type == scheduler_dict.get('name'):
                        self.logger.info('***** Loading scheduler state dict *****')
                        self.scheduler.load_state_dict(scheduler_dict.get('state_dict'))
                    else:
                        self.logger.info('***** Your scheduler is different from the saved *****')

    def load_model(self):
        self.model.cpu()
        self.model = (self.model.module if hasattr(self.model, 'module') else self.model)
        model_path = os.path.join(self.args.training.best_save_path, 'model.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.logger.info('***** Loading checkpoint at %s *****', model_path)
        self.model.to(self.args.model.device)

    def seed_everything(self):
        '''
        设置整个开发环境的seed
        :param seed:
        :param device:
        :return:
        '''
        seed = self.args.training.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    def metric(self, preds, true, data_type):
        acc = metrics.accuracy_score(true, preds)
        f1 = metrics.f1_score(true, preds, average=self.args.metrics.average)
        p = metrics.precision_score(true, preds, average=self.args.metrics.average)
        r = metrics.recall_score(true, preds, average=self.args.metrics.average)
        res = {'acc': acc, 'f1': f1, 'precision': p, 'recall': r}
        if data_type == 'test':
            report = metrics.classification_report(true, preds, digits=4)
            return (res, report)
        return (res,)

    def init_log_writer(self):
        log_dir = self.init_tensorboardx()
        return log_dir

    def step_Plateau_scheduler(self, outputs: torch.Tensor, labels: torch.Tensor, loss: float):
        if self.args.scheduler.metric == 'loss':
            self.scheduler.step(loss)
        else:
            logits = torch.argmax(outputs, dim=1)
            logits = logits.numpy()
            labels = labels.numpy()
            m = self.metric(logits, labels, data_type='train')[0]
            if self.args.scheduler.metric == 'acc':
                self.scheduler.step(m['acc'])
            elif self.args.scheduler.metric == 'f1':
                self.scheduler.step(m['f1'])
            elif self.args.scheduler.metric == 'precision':
                self.scheduler.step(m['p'])
            elif self.args.scheduler.metric == 'recall':
                self.scheduler.step(m['r'])
            else:
                raise TypeError('Plateau LR scheduler has no such type reduce strategy, choose in [{}]'.format(m.keys()))

    def step(self, outputs, labels, loss):
        if self.args.scheduler.type == 'plateau':
            self.step_Plateau_scheduler(outputs, labels, loss)
        else:
            self.scheduler.step()  # Update learning rate schedule
