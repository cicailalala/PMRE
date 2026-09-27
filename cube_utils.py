import numpy as np
import torch


def generate_mask(img, nb_chnls=1):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)

    volume_loc_ind = torch.arange(0, batch_size).cuda()
    volume_loc_ind = volume_loc_ind.view(batch_size, 1, 1, 1).repeat(1, channel, img_x, img_y)
    # partition
    rand_loc_ind = torch.argsort(torch.rand(batch_size, 1, 1), dim=0).cuda()
    cube_part_ind = rand_loc_ind.view(batch_size, 1, 1, 1)
    cube_part_ind = cube_part_ind.repeat_interleave(channel, dim=1)
    cube_part_ind = cube_part_ind.repeat_interleave(patch_x, dim=2)
    cube_part_ind = cube_part_ind.repeat_interleave(patch_y, dim=3)
    volume_loc_ind[:, :, w:w+patch_x, h:h+patch_y] = cube_part_ind
    
    volume_rec_ind = torch.arange(0, batch_size).cuda()
    volume_rec_ind = volume_rec_ind.view(batch_size, 1, 1, 1).repeat(1, nb_chnls, img_x, img_y) 
    # recovery
    rec_ind = torch.argsort(rand_loc_ind, dim=0).cuda()
    cube_rec_ind = rec_ind.view(batch_size, 1, 1, 1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(patch_x, dim=2)
    cube_rec_ind = cube_rec_ind.repeat_interleave(patch_y, dim=3)
    volume_rec_ind[:, :, w:w+patch_x, h:h+patch_y] = cube_rec_ind
    
    return volume_loc_ind.long(), volume_rec_ind.long()


def generate_mask_t(img, nb_chnls=1):
    batch_size, t, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[3]
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)

    volume_loc_ind = torch.arange(0, batch_size).cuda()
    volume_loc_ind = volume_loc_ind.view(batch_size, 1, 1, 1, 1).repeat(1, t, channel, img_x, img_y)
    # partition
    rand_loc_ind = torch.argsort(torch.rand(batch_size, 1, 1), dim=0).cuda()
    cube_part_ind = rand_loc_ind.view(batch_size, 1, 1, 1, 1)
    cube_part_ind = cube_part_ind.repeat_interleave(t, dim=1)
    cube_part_ind = cube_part_ind.repeat_interleave(channel, dim=2)
    cube_part_ind = cube_part_ind.repeat_interleave(patch_x, dim=3)
    cube_part_ind = cube_part_ind.repeat_interleave(patch_y, dim=4)
    volume_loc_ind[:, :, :, w:w+patch_x, h:h+patch_y] = cube_part_ind
    
    volume_rec_ind = torch.arange(0, batch_size).cuda()
    volume_rec_ind = volume_rec_ind.view(batch_size, 1, 1, 1, 1).repeat(1, t, nb_chnls, img_x, img_y) 
    # recovery
    rec_ind = torch.argsort(rand_loc_ind, dim=0).cuda()
    cube_rec_ind = rec_ind.view(batch_size, 1, 1, 1, 1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(t, dim=1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=2)
    cube_rec_ind = cube_rec_ind.repeat_interleave(patch_x, dim=3)
    cube_rec_ind = cube_rec_ind.repeat_interleave(patch_y, dim=4)
    volume_rec_ind[:, :, :, w:w+patch_x, h:h+patch_y] = cube_rec_ind
    
    return volume_loc_ind.long(), volume_rec_ind.long()


