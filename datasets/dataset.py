import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import h5py


def random_rot_flip(image, label, SDF, OSDF):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    SDF = np.rot90(SDF, k)
    OSDF = np.rot90(OSDF, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    SDF = np.flip(SDF, axis=axis).copy()
    OSDF = np.flip(OSDF, axis=axis).copy()
    return image, label, SDF, OSDF


def random_rotate(image, label, SDF, OSDF):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    SDF = ndimage.rotate(SDF, angle, order=0, reshape=False)
    OSDF = ndimage.rotate(OSDF, angle, order=0, reshape=False)
    return image, label,SDF, OSDF


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, SDF, OSDF = sample['image'], sample['label'], sample['SDF'], sample['OSDF']

        # if random.random() > 0.5:
        #     image, label, SDF, OSDF = random_rot_flip(image, label, SDF, OSDF)
        #     image, label, SDF, OSDF = random_rotate(image, label, SDF, OSDF)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            SDF = zoom(SDF, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            OSDF = zoom(OSDF, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        SDF = torch.from_numpy(SDF.astype(np.float32))
        OSDF = torch.from_numpy(OSDF.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'SDF': SDF, 'OSDF': OSDF}
        return sample


class dataset(Dataset):
    def __init__(self,dataset_name, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path,allow_pickle=True)
            image, label, SDF, OSDF = data['image'], data['label'], data['SDF'], data['OSDF']
            sample = {'image': image, 'label': label, 'SDF': SDF, 'OSDF': OSDF}
        else:
            if self.dataset_name == 'LA_MRI' or self.dataset_name == 'Liver_CT':
                vol_name = self.sample_list[idx].strip('\n')
                filepath = self.data_dir + "/{}".format(vol_name) + ".h5"
                data = h5py.File(filepath, 'r')
                image, label = data['image'][:], data['label'][:]
                sample = {'image': image, 'label': label}
            else:
                slice_name = self.sample_list[idx].strip('\n')
                data_path = os.path.join(self.data_dir, slice_name+'.npz')
                data = np.load(data_path,allow_pickle=True)
                image, label = data['image'][:], data['label'][:]
                sample = {'image': image, 'label': label}


        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
