# coding=utf-8
import torch.utils.data as data
import os
import numpy as np
import torch
from .transforms2D import RandomRotate, RandomFlip_LR, RandomFlip_UD, Compose, Resize, RandomCrop2D
import nibabel as nib
from torch.utils.data import DataLoader


def load_dataset_list(dataset, data_root):
    list_path = os.path.join(os.path.dirname(__file__), '..', 'datainfo', dataset + '.txt')
    with open(list_path) as f:
        samples_lists = [os.path.join(data_root, line.strip()) for line in f if line.strip()]
    print(dataset, len(samples_lists))
    return sorted(samples_lists)


def single_frame_dataloader(args):
    if getattr(args, 'inference', False):
        samples = load_dataset_list(args.target_dataset, args.data_root)
        dataset = SingleProstate(args, samples, mode='test')
        return DataLoader(dataset=dataset, batch_size=args.batch_size_single_frame*2,
                          num_workers=args.num_workers, drop_last=False,
                          shuffle=False, pin_memory=True)

    sources = ['RUNMC', 'UCL', 'BIDMC', 'I2CVB', 'BMC', 'HK']
    sources.remove(args.target_dataset)
    samples = []
    for source in sources:
        samples.extend(load_dataset_list(source, args.data_root))
    dataset = SingleProstate(args, samples, mode='train')
    return DataLoader(dataset=dataset, batch_size=args.batch_size_single_frame,
                      num_workers=args.num_workers, drop_last=True,
                      shuffle=True, pin_memory=True)


class SingleProstate(data.Dataset):
    def __init__(self, args, samples_lists, mode='Train'):
        self.mode = mode
        self.args = args
        self.imgs = []
        self.case_slice_ids = []
        for sample_path in samples_lists:
            dataset = sample_path.split('/')[-2]
            sample = nib.load(sample_path)
            images = sample.dataobj
            images = np.array(images).transpose((2, 0, 1))
            gt_path = sample_path.replace('.nii.gz', '_segmentation.nii.gz')
            if not os.path.exists(gt_path):
                gt_path = sample_path.replace('.nii.gz', '_Segmentation.nii.gz')
            sample_gt = nib.load(gt_path)
            masks = sample_gt.dataobj
            masks = np.array(masks).transpose((2, 0, 1))
            n = images.shape[0]
            
            # images = images.clip(min=-125, max=275).astype(int)

            binary_mask = np.ones(masks.shape)
            mean = np.sum(images * binary_mask) / np.sum(binary_mask)
            std = np.sqrt(np.sum(np.square(images - mean) * binary_mask) / np.sum(binary_mask)) 
            images = (images - mean) / std 
            # normalize per image, using statistics within the brain, but apply to whole image
            masks[masks==2] = 1


            for i in range(n):
                image = images[i, :, :][np.newaxis, :, :]
                mask = masks[i, :, :][np.newaxis, :, :]
                # image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
                # mask = Image.fromarray(mask.astype(np.uint8))
                '''
                if mode != 'train':
                    if np.sum(mask) > 0:
                        self.imgs.append((image, mask))
                        self.case_slice_ids.append(dataset + '_' + sample_path.split('/')[-1].split('.')[0] + '_' + str(i).zfill(2))
                else:
                    self.imgs.append((image, mask))
                    self.case_slice_ids.append(dataset + '_' + sample_path.split('/')[-1].split('.')[0] + '_' + str(i).zfill(2))
                '''
                if np.sum(mask) > 0:
                    self.imgs.append((image, mask))
                    self.case_slice_ids.append(dataset + '_' + sample_path.split('/')[-1].split('.')[0] + '_' + str(i).zfill(2))                


        size = self.args.resolution
        if self.mode == 'train':
            self.transform = Compose([
                Resize((int(size[0]*1.1), int(size[1]*1.1))),
                RandomCrop2D(size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate(15),
                # Normalize(mean=self.args.mean, std=self.args.std)
            ])
        else:
            self.transform = Compose([
                Resize(self.args.resolution),
                # Normalize(mean=self.args.mean, std=self.args.std)
            ])

    def __getitem__(self, item):
        image, mask = self.imgs[item]
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        h, w = image.size(1), image.size(2)
        size = (h, w)

        lam = np.random.beta(self.args.Mixupalpha, self.args.Mixupalpha)
        if self.args.Mixup and self.mode == 'train':
            idx = np.random.choice(self.__len__())
            Miximage, Mixmask = self.imgs[idx]
            Miximage = torch.from_numpy(Miximage)
            Mixmask = torch.from_numpy(Mixmask)
            image = (lam * image + (1 - lam) * Miximage)
            if self.transform:
                image, mask, Mixmask = self.transform(image, mask, Mixmask)
            Mixmask[Mixmask < 0.5] = 0
            Mixmask[Mixmask >= 0.5] = 1
            Mixmask = Mixmask.squeeze(0).long()
        else:
            if self.transform:
                image, mask = self.transform(image, mask)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        image = image.squeeze(0)
        image = image.repeat(3, 1, 1)
        mask = mask.squeeze(0).long()
        if self.args.Mixup and self.mode == 'train':
            sample = {'image': image, 'mask': mask, 'Mixmask': Mixmask}
            sample['lam'] = torch.from_numpy(np.array(lam))
        else:
            sample = {'image': image, 'mask': mask}

        if self.mode != 'train':
            case_path = self.case_slice_ids[item]
            case_slice_id = os.path.basename(case_path)
            sample['case_slice_id'] = case_slice_id
            sample['size'] = torch.tensor(size)

        return sample


    def __len__(self):
        return len(self.imgs)

    # testloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
