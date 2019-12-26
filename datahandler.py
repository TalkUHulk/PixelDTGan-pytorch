import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image


class DataHandler(Dataset):
    def __init__(self, tran_data_dir, patch_size=64, augment=False):
        self.extensions = ('.png', '.jpeg', '.jpg')
        self.patch_size = patch_size
        self.augment = augment
        data_tmp = self._get_train_set(tran_data_dir)
        self.target_files = data_tmp[1]
        self.image_files = data_tmp[0]
        self.transforms = transforms.RandomHorizontalFlip(p=0.5)


    def __getitem__(self, index):
        target_prefix = self.image_files[index].split('/')[-1].split('_')[0]

        nopaired = np.random.choice(self.image_files, 1)[0]

        while target_prefix == nopaired.split('/')[-1].split('_')[0]:
            nopaired = np.random.choice(self.image_files, 1)[0]

        np_target_prefix = nopaired.split('/')[-1].split('_')[0]

        image = Image.open(self.image_files[index])
        target = Image.open(self.target_files[target_prefix])
        np_target = Image.open(self.target_files[np_target_prefix])

        if self.augment:
            image = self.transforms(image)

        return self._pre_process(image), self._pre_process(target), self._pre_process(np_target)



    def __len__(self):
        return len(self.image_files)

    def _pre_process(self, img):
        w, h = img.size
        h_n = self.patch_size if h >= w else int(h / (w / self.patch_size))
        w_n = self.patch_size if w >= h else int(w / (h / self.patch_size))

        img = img.resize((w_n, h_n), Image.BICUBIC)
        if h_n != self.patch_size:
            pad_left = (self.patch_size - h_n) // 2
            pad_right = self.patch_size - h_n - pad_left
            img = np.pad(img, ((pad_left, pad_right), (0, 0), (0, 0)), mode='constant',
                         constant_values=255)

        if w_n != self.patch_size:
            pad_left = (self.patch_size - w_n) // 2
            pad_right = self.patch_size - w_n - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant',
                         constant_values=255)

        #img = img / 255.0

        return torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0 * 2.0 - 1.0).type(torch.FloatTensor)    # H x W x C --> C x H x W

    def _get_train_set(self, path):
        file_names = [x.path for x in os.scandir(path)
                      if x.name.endswith(self.extensions) and 'CLEAN0' in x.name]
        target_names = [x.path for x in os.scandir(path)
                        if x.name.endswith(self.extensions) and 'CLEAN1' in x.name]
        pid = [p.split('/')[-1].split('_')[0] for p in target_names]

        target_files = {}
        for i in range(len(target_names)):
            tmp = {pid[i]: target_names[i]}
            target_files.update(tmp)

        return [np.sort(file_names), target_files]








