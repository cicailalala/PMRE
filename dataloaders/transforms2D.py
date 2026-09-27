import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms


# ----------------------data augment-------------------------------------------
class Resize:
    def __init__(self, size):

        self.size = size

    def __call__(self, img, mask, Mixmask=None):
        T = img.size(0)
        size = [T, self.size[0], self.size[1]]
        img, mask = img.unsqueeze(0).unsqueeze(0).float(), mask.unsqueeze(0).unsqueeze(0).float()
        img = F.interpolate(img, size=size, mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=size, mode='trilinear', align_corners=False)
        if Mixmask is not None:
            Mixmask = Mixmask.unsqueeze(0).unsqueeze(0).float()
            Mixmask = F.interpolate(Mixmask, size=size, mode='trilinear', align_corners=False)
            return img[0], mask[0], Mixmask[0]
        else:
            return img[0], mask[0]


class RandomCrop2D:
    def __init__(self, size):
        self.size = size  # h, w

    def __call__(self, img, mask, Mixmask=None):
        h, w = img.size(2), img.size(3)
        th, tw = self.size  # target size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)


        tmp_img = img[:, :, x1:x1 + th, y1:y1 + tw]
        tmp_mask = mask[:, :, x1:x1 + th, y1:y1 + tw]

        if Mixmask is not None:
            tmp_Mixmask = Mixmask[:, :, x1:x1 + th, y1:y1 + tw]
            return tmp_img, tmp_mask, tmp_Mixmask
        else:
            return tmp_img, tmp_mask


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask, Mixmask=None):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        if Mixmask is not None:
            return self._flip(img, prob), self._flip(mask, prob), self._flip(Mixmask, prob)
        else:
            return self._flip(img, prob), self._flip(mask, prob)


class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask, Mixmask=None):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        if Mixmask is not None:
            return self._flip(img, prob), self._flip(mask, prob), self._flip(Mixmask, prob)
        else:
            return self._flip(img, prob), self._flip(mask, prob)


class RandomRotate:
    def __init__(self, max_cnt=360):
        self.max_cnt = max_cnt

    def get_rot_mat(self, theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def rot_img(self, x, theta, dtype):
        rot_mat = self.get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size(), align_corners=True).type(dtype)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def __call__(self, img, mask, Mixmask=None):
        # im should be a 4D tensor of shape B x C x H x W with type dtype, range [0,255]
        cnt = random.randint(0, self.max_cnt*2) - self.max_cnt
        dtype = torch.FloatTensor
        theta = cnt / 180 * np.pi
        if Mixmask is not None:
            return self.rot_img(img, theta, dtype), self.rot_img(mask, theta, dtype), self.rot_img(Mixmask, theta, dtype)
        else:
            return self.rot_img(img, theta, dtype), self.rot_img(mask, theta, dtype)

# class RandomRotate:
#     def __init__(self, max_cnt=3):

#
#     def _rotate(self, img, cnt):
#         img = torch.rot90(img, cnt, [2, 3])
#         return img
#
#     def __call__(self, img, mask):
#         cnt = random.randint(0, self.max_cnt)
#         return self._rotate(img, cnt), self._rotate(mask, cnt)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, Mixmask=None):
        if Mixmask is not None:
            for t in self.transforms:
                img, mask, Mixmask = t(img, mask, Mixmask)
            return img, mask, Mixmask
        else:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask