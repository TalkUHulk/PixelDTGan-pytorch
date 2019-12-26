import torch
import torch.utils.data as Data
import torchvision.utils as vutils
from datahandler import DataHandler
from torchvision.utils import save_image
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2 as cv



class Tester:
    def __init__(
        self,
        generator,
        test_dir='./data/test/',
        result_dir='./result_1/',
        weight_dir='./weight/generator_epoch_999.pkl',
        batch_size=32,
        patch_size=64,
        num_workers=8,
        self_dir=None,
        self_test=False,
        cuda=True,
        extensions=('.png', '.jpeg', '.jpg')
    ):
        self.patch_size = patch_size
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        self.generator = generator
        self.weight_dir = weight_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
        self.generator.load_state_dict(torch.load(self.weight_dir, map_location=self.device))
        self.generator.eval()
        self.self_test = self_test
        if self.self_test:
            self.self_dir = self_dir
            self.extensions = extensions
            self.test_file = [x.path for x in os.scandir(self.self_dir) if x.name.endswith(self.extensions)]
        else:
            self.num_workers = num_workers
            self.batch_size = batch_size
            self.test_dh = DataHandler(test_dir, patch_size=self.patch_size, augment=False)
            self.test_loader = Data.DataLoader(
                self.test_dh, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


    def process_output(self, input):
        return (input + 1.0) / 2.0


    def test(self):
        with torch.no_grad():

            with tqdm(desc='Testing',
                      total=len(self.test_file) if self.self_test else len(self.test_loader)) as pbar:
                if self.self_test:
                    for i, data_batch in enumerate(self.test_file):
                        input, hw = self._pre_process(data_batch)
                        output = self.generator.forward(input)

                        output = self._out_process(output)
                        output = cv.cvtColor(output, cv.COLOR_BGR2RGB)
                        cv.imwrite(
                            os.path.join(self.result_dir, "{}_result.png".format(data_batch.split('/')[-1].split('.')[0])),
                                   output)

                        pbar.update()
                else:
                    for i, data_batch in enumerate(self.test_loader):
                        source_batch, target_batch, _ = data_batch
                        source_batch = source_batch.to(self.device)
                        target_batch = target_batch.to(self.device)

                        gen_batch = self.generator(source_batch)

                        gen_batch = self.process_output(gen_batch)  # [-1, 1]->[0, 1]
                        source_batch = self.process_output(source_batch)
                        target_batch = self.process_output(target_batch)

                        concat = torch.cat([source_batch, target_batch, gen_batch], 3)
                        x = vutils.make_grid(concat, normalize=True, scale_each=True)

                        save_image(x, os.path.join(self.result_dir, "Test_{}.png".format(i)), scale_each=True)

                        pbar.update()

    def _pre_process(self, img_path):
        img = Image.open(img_path)
        h, w = img.size

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


        return torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0 * 2.0 - 1.0).unsqueeze(0).type(
            torch.FloatTensor), [h, w]  # H x W x C --> C x H x W


    def _out_process(self, tensor):

        output = tensor.squeeze(0).numpy()
        output = np.uint8(np.clip((output + 1.0) / 2.0 * 255 + .5, 0, 255))
        output = np.transpose(output, (1, 2, 0))
        return output












