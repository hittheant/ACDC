from os.path import splitext
from os import listdir
import os
import fnmatch
import random
import numpy as np
import nibabel as nib
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class Project1(Dataset):

    def __init__(self, data_dir):
        super().__init__()
        image_path = []
        patient_paths = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        for patient in patient_paths:
            info = read_info(data_dir + patient + '/Info.cfg')
            ed = data_dir + patient + '/' + patient + '_frame' + str(
                info['ED']).zfill(2)
            es = data_dir + patient + '/' + patient + '_frame' + str(
                info['ES']).zfill(2)
            img = nib.load(ed + '.nii.gz')
            for i in range(img.get_fdata().shape[2]):
                image_path.append([ed + '.nii.gz', ed + '_gt.nii.gz', i])
            for i in range(img.get_fdata().shape[2]):
                image_path.append([es + '.nii.gz', es + '_gt.nii.gz', i])

        random.shuffle(image_path)
        print(image_path)
        assert len(image_path) > 0

        self.image_path = image_path
        self.transform = [transforms.ToTensor(), transforms.Resize((256, 256))]
        self.transform = transforms.Compose(self.transform)

    def __getitem__(self, item):
        image = nib.load(self.image_path[item][0])
        gt = nib.load(self.image_path[item][1])
        image = image.get_fdata()[:, :, self.image_path[item][2]].squeeze()
        gt = gt.get_fdata()[:, :, self.image_path[item][2]].squeeze()
        image = self.transform(image)
        gt = self.transform(gt)
        return {
            'image': image,
            'mask': gt.squeeze()
        }

    def __len__(self):
        return len(self.image_path)

class Project2(Dataset):

    def __init__(self, data_dir):
        super().__init__()
        image_path = []
        patient_paths = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        for patient in patient_paths:
            patient_dir = data_dir + patient + '/' + patient

            for i in range(155):
                image_path.append([patient_dir, i])

        random.shuffle(image_path)
        print(image_path)
        assert len(image_path) > 0

        self.image_path = image_path
        self.transform = transforms.ToTensor()

    def __getitem__(self, item):
        patient_path = self.image_path[item][0]
        i = self.image_path[item][1]
        flair = nib.load('{0}_flair.nii.gz'.format(patient_path))
        flair = flair.get_fdata()
        t1 = nib.load('{0}_flair.nii.gz'.format(patient_path))
        t1 = t1.get_fdata()
        t1ce = nib.load('{0}_flair.nii.gz'.format(patient_path))
        t1ce = t1ce.get_fdata()
        t2 = nib.load('{0}_flair.nii.gz'.format(patient_path))
        t2 = t2.get_fdata()
        gt = nib.load('{0}_seg.nii.gz'.format(patient_path))
        image = np.concatenate((np.expand_dims(flair[:, :, i], axis=0),
                                np.expand_dims(t1[:, :, i], axis=0),
                                np.expand_dims(t1ce[:, :, i], axis=0),
                                np.expand_dims(t2[:, :, i], axis=0)), axis=0)
        gt = gt.get_fdata()[:, :, i].squeeze()
        gt = self.transform(gt)
        return {
            'image': image,
            'mask': gt.squeeze()
        }

    def __len__(self):
        return len(self.image_path)

class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# Return dict with keys: 'ED', 'ES', 'Group', 'Height', 'NbFrame', 'Weight'
def read_info(path):
    f = open(path, 'r')
    Lines = f.readlines()
    info = defaultdict(list)
    for line in Lines:
        res = dict(map(lambda x: x.split(': '), line.rstrip("\n").split(",")))
        info.update(res)
    return info