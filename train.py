import argparse
import torch
import os
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from dataloaders import prostate_single_frame, prostate_multi_frames
from dataloaders.utils import SoftDiceLoss, MetricMonitor, BCE_2D
from model import MattingNetwork
import cube_utils

class Trainer:
    def __init__(self):
        self.parse_args()
        self.init_datasets()
        self.init_save_dir()
        self.init_model()
        self.train()
        fo = open(self.save_dir + '/' + self.args.target_dataset + '_log.txt', 'a')
        fo.writelines(self.save_infos)
        fo.close()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-root', type=str, required=True, help='Directory containing the six domain folders')
        # Model
        parser.add_argument('--model-variant', type=str, default='ConvResNet', choices=['ConvResNet'])

        # Learning rate
        parser.add_argument('--learning-rate-backbone', type=float, default=0.0002)
        parser.add_argument('--learning-rate-aspp', type=float, default=0.0004)
        parser.add_argument('--learning-rate-decoder', type=float, default=0.0004)
        # Training setting
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--target_dataset', type=str, default='UCL', choices=['RUNMC', 'BMC', 'I2CVB', 'UCL', 'BIDMC', 'HK'])
        parser.add_argument('--resolution', default=(384, 384))
        parser.add_argument('--train-single', type=str, default='multi', choices=['single', 'multi'])
        parser.add_argument('--batch-size-single-frame', type=int, default=5)
        parser.add_argument('--sequence-len', type=int, default=4)
        parser.add_argument('--batch-size-multi-frames', type=int, default=2)
        parser.add_argument('--num-workers', type=int, default=2)
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=40)

        parser.add_argument('--SegVREx', type=int, default=True)
        parser.add_argument('--Mixup', type=int, default=True)
        parser.add_argument('--Mixupalpha', type=float, default=0.2)

        parser.add_argument('--save-dir', type=str, required=True)
        parser.add_argument('--log-interval', type=int, default=30)
        # Checkpoint loading and saving
        parser.add_argument('--checkpoint', type=str)

        # Debugging
        parser.add_argument('--disable-mixed-precision', action='store_true')
        self.args = parser.parse_args()
        self.args.save_dir = os.path.join(self.args.save_dir, '')
        if self.args.epoch_start < 0 or self.args.epoch_end <= self.args.epoch_start:
            parser.error('Require 0 <= epoch-start < epoch-end.')
        print(self.args)

    def init_datasets(self):
        if self.args.train_single == 'single':
            self.dataloader_single_frame_train = prostate_single_frame.single_frame_dataloader(self.args)
        else:
            self.dataloader_multi_frames_train = prostate_multi_frames.multi_frames_dataloader(self.args)

    def init_model(self):
        print('Initializing model')
        self.model = MattingNetwork(self.args.model_variant, classes= self.args.num_classes, pretrained_backbone=False)

        if self.args.checkpoint:
            load_path = os.path.join(self.save_dir, self.args.checkpoint)
            print(f'Restoring from checkpoint: ', load_path)
            print(self.model.load_state_dict(torch.load(load_path)))
        self.model = self.model.cuda()
        self.DiceLoss_fn = SoftDiceLoss(reduction='none').cuda()
        self.BCE_fn = BCE_2D(reduction='none').cuda()

        self.optimizer = Adam([
            {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
            {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp},
            {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.project_seg.parameters(), 'lr': self.args.learning_rate_decoder},
        ])
        self.scaler = GradScaler()

    def init_save_dir(self):
        print('Initializing save dir')
        save_name = self.args.target_dataset + '_' + self.args.model_variant + '_' + str(self.args.resolution[0]) + str(self.args.resolution[1])
        save_name = save_name + '_' + str(self.args.batch_size_single_frame)
        self.save_dir = os.path.join(self.args.save_dir, save_name)
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_infos = []

    def train(self):
        if self.args.train_single == 'single':
            loader = self.dataloader_single_frame_train
            train_epoch = self.train_single_frame
        else:
            loader = self.dataloader_multi_frames_train
            train_epoch = self.train_multi_frames
        if len(loader) == 0:
            raise ValueError('No training batches; check the dataset and batch size.')
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch
            self.step = epoch * len(loader)
            train_epoch(epoch - self.args.epoch_start)
        self.save()

    def train_single_frame(self, i_epoch):
        self.model.train()
        metric_monitor = MetricMonitor()
        stream = tqdm(self.dataloader_single_frame_train)

        max_iterations = len(self.dataloader_single_frame_train) * (self.args.epoch_end - self.args.epoch_start)
        
        for i, sample_batched in enumerate(stream, start=1):
            images, target = sample_batched['image'], sample_batched['mask']
            true_img = images.cuda()
            true_seg = target.cuda()

            # Cross-image Partition-and-Recovery
            bs, c, w, h = true_img.shape # (5, 3, 384, 384)

            cube_part_ind, cube_rec_ind = cube_utils.generate_mask(true_img, nb_chnls=self.args.num_classes)

            img_cross_mix = true_img.view(bs, c, w, h)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, c, w, h)

            with autocast(enabled=not self.args.disable_mixed_precision):
                mix_img = torch.cat((true_img, img_cross_mix), dim=0)
                mix_seg = self.model(mix_img)[0]
                pred_seg, cross_seg = mix_seg[:bs], mix_seg[bs:]
                cross_seg = torch.gather(cross_seg, dim=0, index=cube_rec_ind)
                mix_seg = torch.cat((pred_seg, cross_seg), dim=0)
                true_seg = torch.cat((true_seg, true_seg), dim=0)

                diceL = self.DiceLoss_fn(mix_seg, true_seg)
                ceL = self.BCE_fn(mix_seg, true_seg)

                if self.args.Mixup:
                    Mixtrue_seg, lam = sample_batched['Mixmask'].cuda(), sample_batched['lam'].cuda()
                    Mixtrue_seg = torch.cat((Mixtrue_seg, Mixtrue_seg), dim=0)
                    MixdiceL = self.DiceLoss_fn(mix_seg, Mixtrue_seg)
                    MixceL = self.BCE_fn(mix_seg, Mixtrue_seg)
                    lam = torch.cat((lam, lam), dim=0)
                    lam = lam.unsqueeze(1)
                    diceL = lam * diceL + (1 - lam) * MixdiceL
                    ceL = lam * ceL + (1 - lam) * MixceL

                meandiceL = diceL.mean()
                meanceL = ceL.mean()

                penalty = ((diceL - meandiceL) ** 2).mean() + ((ceL - meanceL) ** 2).mean()
                if self.args.SegVREx:
                    loss = meandiceL + meanceL + penalty
                else:
                    loss = meandiceL + meanceL

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            metric_monitor.update("Loss", loss.item(), images.size(0))
            metric_monitor.update("DiceL", meandiceL.item(), images.size(0))
            metric_monitor.update("penalty", penalty.item(), images.size(0))
            stream.set_description(
                "Epoch: {epoch}. Train. {metric_monitor}".format(epoch=self.epoch, metric_monitor=metric_monitor)
            )
            if self.step % self.args.log_interval == 0:
                self.save_infos.append('TrainEpoch %2d, trainingLoss %.4f DiceLoss %.4f penalty %.4f\n' % (self.epoch, loss.item(), meandiceL.item(), penalty.item()))
            self.step += 1

            for param_group in self.optimizer.param_groups:
                iter_n = len(self.dataloader_single_frame_train)*i_epoch+i
                base_lr = param_group['lr'] / ((1.0 - (iter_n-1) / max_iterations) ** 0.9)
                lr_ = base_lr * (1.0 - iter_n / max_iterations) ** 0.9
                param_group['lr'] = lr_ 
                if iter_n % 100 == 0:
                    print('lr', max_iterations, base_lr, iter_n, lr_)
        trainLoss = metric_monitor.metrics['Loss']['avg']
        trainDiceLoss = metric_monitor.metrics['DiceL']['avg']
        trainpenalty = metric_monitor.metrics['penalty']['avg']
        print('TrainEpoch: {}, trainLoss: {:.4f}, trainDiceLoss: {:.4f}, penalty: {:.4f}'.format(self.epoch, trainLoss, trainDiceLoss, penalty))
        self.save_infos.append('TrainEpoch: {}, trainLoss: {:.4f}, trainDiceLoss: {:.4f}, penalty: {:.4f}\n'.format(self.epoch, trainLoss, trainDiceLoss, trainpenalty))

    def train_multi_frames(self, i_epoch):
        self.model.train()
        metric_monitor = MetricMonitor()
        stream = tqdm(self.dataloader_multi_frames_train)

        max_iterations = len(self.dataloader_multi_frames_train) * (self.args.epoch_end - self.args.epoch_start)
        for i, sample_batched in enumerate(stream, start=1):
            images, target = sample_batched['image'], sample_batched['mask']
            true_img = images.cuda()
            true_seg = target.cuda()

            # Cross-image Partition-and-Recovery
            bs, t, c, w, h = true_img.shape # (2, 8, 3, 384, 384)

            cube_part_ind, cube_rec_ind = cube_utils.generate_mask_t(true_img, nb_chnls=self.args.num_classes)

            img_cross_mix = true_img.view(bs, t, c, w, h)
            img_cross_mix = torch.gather(img_cross_mix, dim=0, index=cube_part_ind)
            img_cross_mix = img_cross_mix.view(bs, t, c, w, h)

            with autocast(enabled=not self.args.disable_mixed_precision):
                mix_img = torch.cat((true_img, img_cross_mix), dim=0)
                mix_seg = self.model(mix_img)[0]
                pred_seg, cross_seg = mix_seg[:bs], mix_seg[bs:]
                cross_seg = torch.gather(cross_seg, dim=0, index=cube_rec_ind)
                mix_seg = torch.cat((pred_seg, cross_seg), dim=0)
                true_seg = torch.cat((true_seg, true_seg), dim=0) 

                mix_seg = mix_seg.permute(0, 2, 1, 3, 4).contiguous()
                true_seg = true_seg.permute(0, 2, 1, 3, 4).contiguous()                               
                
                diceL = self.DiceLoss_fn(mix_seg, true_seg)
                ceL = self.BCE_fn(mix_seg, true_seg)                

                if self.args.Mixup:
                    Mixtrue_seg, lam = sample_batched['Mixmask'].cuda(), sample_batched['lam'].cuda()
                    Mixtrue_seg = torch.cat((Mixtrue_seg, Mixtrue_seg), dim=0)
                    Mixtrue_seg = Mixtrue_seg.permute(0, 2, 1, 3, 4).contiguous()
                    MixdiceL = self.DiceLoss_fn(mix_seg, Mixtrue_seg)
                    MixceL = self.BCE_fn(mix_seg, Mixtrue_seg)
                    lam = torch.cat((lam, lam), dim=0)
                    lam = lam.unsqueeze(1)
                    diceL = lam * diceL + (1 - lam) * MixdiceL
                    ceL = lam * ceL + (1 - lam) * MixceL

                meandiceL = diceL.mean()
                meanceL = ceL.mean()

                penalty = ((diceL - meandiceL) ** 2).mean() + ((ceL - meanceL) ** 2).mean()
                if self.args.SegVREx:
                    loss = meandiceL + meanceL + penalty
                else:
                    loss = meandiceL + meanceL

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            metric_monitor.update("Loss", loss.item(), images.size(0))
            metric_monitor.update("DiceL", meandiceL.item(), images.size(0))
            metric_monitor.update("penalty", penalty.item(), images.size(0))
            stream.set_description(
                "Epoch: {epoch}. Train. {metric_monitor}".format(epoch=self.epoch, metric_monitor=metric_monitor)
            )
            if self.step % self.args.log_interval == 0:
                self.save_infos.append('Epoch %2d, trainingLoss %.4f DiceLoss %.4f penalty %.4f\n' % (self.epoch, loss.item(), meandiceL.item(), penalty.item()))
            self.step += 1

            for param_group in self.optimizer.param_groups:
                iter_n = len(self.dataloader_multi_frames_train)*i_epoch+i
                base_lr = param_group['lr'] / ((1.0 - (iter_n-1) / max_iterations) ** 0.9)
                lr_ = base_lr * (1.0 - iter_n / max_iterations) ** 0.9
                param_group['lr'] = lr_ 
                if iter_n % 100 == 0:
                    print('lr', max_iterations, base_lr, iter_n, lr_)

        trainLoss = metric_monitor.metrics['Loss']['avg']
        trainDiceLoss = metric_monitor.metrics['DiceL']['avg']
        trainpenalty = metric_monitor.metrics['penalty']['avg']
        print('TrainEpoch: {}, trainLoss: {:.4f}, trainDiceLoss: {:.4f}, trainpenalty: {:.4f}'.format(self.epoch, trainLoss, trainDiceLoss, trainpenalty))
        self.save_infos.append('TrainEpoch: {}, trainLoss: {:.4f}, trainDiceLoss: {:.4f}, trainpenalty: {:.4f}\n'.format(self.epoch, trainLoss, trainDiceLoss, trainpenalty))

    def save(self):
        filename = f'{self.args.train_single}_epoch{self.epoch + 1}_last.pth'
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f'Saved final stage checkpoint: {path}')

if __name__ == '__main__':

    Trainer()
