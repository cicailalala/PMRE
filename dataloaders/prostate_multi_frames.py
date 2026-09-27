import torch.utils.data as data
import os
import numpy as np
import torch

from .transforms import RandomRotate, RandomFlip_LR, RandomFlip_UD, Compose, Resize, RandomFlip_T, RandomCrop2D
import nibabel as nib
from torch.utils.data import DataLoader
from .prostate_single_frame import load_dataset_list

def split(samples, squence_len, stride):
    imgs = []
    for key in samples.keys():
        image, mask = samples[key]
        T = image.shape[0]
        n = int((T-squence_len)/stride) + 1
        for i in range(n):
            imgs.append([key, i*stride, i*stride+squence_len])
    return imgs

def multi_frames_dataloader(args):
    if getattr(args, 'inference', False):
        samples = load_dataset_list(args.target_dataset, args.data_root)
        dataset = MultiProstate(args, samples, mode='test')
        return DataLoader(dataset=dataset, batch_size=args.batch_size_multi_frames*2,
                          num_workers=args.num_workers, drop_last=False,
                          shuffle=False, pin_memory=True)

    sources = ['RUNMC', 'UCL', 'BIDMC', 'I2CVB', 'BMC', 'HK']
    sources.remove(args.target_dataset)
    samples = []
    for source in sources:
        samples.extend(load_dataset_list(source, args.data_root))
    dataset = MultiProstate(args, samples, mode='train')
    return DataLoader(dataset=dataset, batch_size=args.batch_size_multi_frames,
                      num_workers=args.num_workers, drop_last=True,
                      shuffle=True, pin_memory=True)


class MultiProstate(data.Dataset):
    def __init__(self, args, samples_lists, mode='train', squence_len=4, stride=1):
        self.mode = mode
        self.args = args
        self.sequence_len = args.sequence_len
        self.stride = stride
        self.samples = {}
        self.cases_infos = {}
        for sample_path in samples_lists:
            dataset = sample_path.split('/')[-2]
            sample = nib.load(sample_path)
            spacing = sample.header.get_zooms()
            image = sample.dataobj
            image = np.array(image).transpose((2, 0, 1))
            gt_path = sample_path.replace('.nii.gz', '_segmentation.nii.gz')
            if not os.path.exists(gt_path):
                gt_path = sample_path.replace('.nii.gz', '_Segmentation.nii.gz')
            sample_gt = nib.load(gt_path)
            mask = sample_gt.dataobj
            mask = np.array(mask).transpose((2, 0, 1))

            n = mask.shape[0]
            binary_mask = np.ones(mask.shape)
            mean = np.sum(image * binary_mask) / np.sum(binary_mask)
            std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask)) 
            image = (image - mean) / std 
            # normalize per image, using statistics within the brain, but apply to whole image
            mask[mask==2] = 1
            
            start = 0
            end = 0
            for i in range(n):
                msk = mask[i]
                if i < n - 1 and np.sum(msk) == 0 and np.sum(mask[i + 1]) > 0:
                    start = i + 1
                if np.sum(msk) > 0:
                    end = i

            if mode == 'train':
                image = image[start: end + 1]
                mask = mask[start: end + 1]
            if mode == 'test':
                image = image[start: end + 1]
                mask = mask[start: end + 1]

            
            case_name = dataset + '_' + sample_path.split('/')[-1].split('.')[0]

            self.samples[case_name] = [image, mask]
            self.cases_infos[case_name] = [case_name, spacing]

        imgs = split(self.samples, self.sequence_len, stride)
        self.imgs = imgs
        size = self.args.resolution
        if self.mode == 'train':
            self.transforms = Compose([
                Resize((int(size[0]*1.1), int(size[1]*1.1))),
                RandomCrop2D(size),
                RandomFlip_T(prob=0.5),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate(),
                # Normalize(mean=self.args.mean, std=self.args.std)
            ])
        else:
            self.transforms = Compose([
                Resize(self.args.resolution),
                # Normalize(mean=self.args.mean, std=self.args.std)
            ])


    def __getitem__(self, item):

        cases_name, start, end = self.imgs[item]
        image = self.samples[cases_name][0][start:end, :, :]
        mask = self.samples[cases_name][1][start:end, :, :]
        image = torch.from_numpy(image)
        mask =torch.from_numpy(mask)

        h, w = image.size(1), image.size(2)

        size = self.samples[cases_name][0].shape[0]

        lam = np.random.beta(self.args.Mixupalpha, self.args.Mixupalpha)
        if self.args.Mixup and self.mode == 'train':
            idx = np.random.choice(self.__len__())
            Mixcases_name, Mixstart, Mixend = self.imgs[idx]
            Miximage = self.samples[Mixcases_name][0][Mixstart:Mixend, :, :]
            Mixmask = self.samples[Mixcases_name][1][Mixstart:Mixend, :, :]
            Miximage = torch.from_numpy(Miximage)
            Mixmask = torch.from_numpy(Mixmask)
            image = (lam * image + (1 - lam) * Miximage)
            if self.transforms:
                image, mask, Mixmask = self.transforms(image, mask, Mixmask)
            Mixmask[Mixmask < 0.5] = 0
            Mixmask[Mixmask >= 0.5] = 1
            Mixmask = Mixmask.squeeze(0).unsqueeze(1).long()
        else:
            if self.transforms:
                image, mask, _ = self.transforms(image, mask, mask)


        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        image = image.squeeze(0).unsqueeze(1)
        image = image.repeat(1, 3, 1, 1)
        mask = mask.squeeze(0).unsqueeze(1)


        if self.args.Mixup and self.mode == 'train':
            sample = {'image': image, 'mask': mask, 'Mixmask': Mixmask}
            sample['lam'] = torch.from_numpy(np.array(lam))
        else:
            sample = {'image': image, 'mask': mask}

        if self.mode != 'train':
            case_path = self.cases_infos[cases_name][0]
            spacing = np.array(self.cases_infos[cases_name][1])
            sample['case_info'] = [case_path, start, end, spacing]
            sample['len'] = torch.tensor(size)


        return sample

    def __len__(self):
        return len(self.imgs)

