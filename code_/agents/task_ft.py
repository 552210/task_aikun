import torch
import random
from .trainer import CLTrainer
import torch.distributed as dist



class TFT(CLTrainer):
    def __init__(self, config, args, logger, out_dim, ckpt_path=None):
        super().__init__(config, args, logger, out_dim, ckpt_path)


        # self.valid_out_dim = str(self.args.task_count)

    def init_optimizer(self):

        last_layer_params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer_arg = {'params': last_layer_params,
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=self.config['gamma'])

    def train_model(self, train_loader, val_loader):
        for epoch in range(self.config['schedule'][-1]):
            # Config the model and optimizer
            # if self.args.is_main_process:
            self.log.info('Epoch:{0}'.format(epoch))
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)

            if len(self.config['out_dim']) > 1:
                currt_cls_name = 'last'+'.'+str(self.args.task_count) +'.weight'
                # for name, param in self.model.named_parameters():
                #     #只冻结其他任务分类器
                #     if name.startswith('last') and name != currt_cls_name:
                #         param.requires_grad = False
                #         else:
                #         param.requires_grad = True
                        #  冻结所有参数

                for name, param in self.model.named_parameters():
                    if name.startswith('last') and name == currt_cls_name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            else:

                for name, param in self.model.named_parameters():
                    if name.startswith('last'):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


            self.scheduler.step(epoch)
            # if self.args.is_main_process:
            for param_group in self.optimizer.param_groups:
                self.log.info('LR:{}'.format(param_group['lr']))

            if self.args.is_main_process:
                log_str = ' Itr\t    Time  \t    Data  \t  Loss  \t  Acc'
            if self.args.distributed:
                log_str = 'Rank\t' + log_str
            self.log.info(log_str)

            super(TFT,self).before_epoch()
            super(TFT,self).train_epoch(train_loader)

            if self.args.distributed:
                dist.barrier()

            super(TFT,self).after_epoch()
            # Evaluate the performance of current task
            if val_loader != None:
                super(TFT,self).validation(val_loader)



