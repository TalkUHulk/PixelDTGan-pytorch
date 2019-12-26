import torch
import torch.utils.data as Data
import torchvision.utils as vutils
import torch.nn as nn
from datahandler import DataHandler
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import logging
import os
import datetime
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(
        self,
        generator,
        discriminatorR,
        discriminatorA,
        train_dir,
        val_dir,
        log_dir='./log',
        weight_dir='./weight',
        learn_rate=0.0002,
        decay_steps=25,
        batch_size=128,
        patch_size=64,
        num_workers=8,
        max_n_weights=5,
        adam_optimizer={'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
        cuda=True
    ):

        self.generator = generator
        self.discriminatorR = discriminatorR
        self.discriminatorA = discriminatorA
        self.log_dir = log_dir
        self.weight_dir = weight_dir
        self.max_n_weights = max_n_weights
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.weight_dir):
            os.mkdir(self.weight_dir)

        self.log_img = os.path.join(self.log_dir, 'images')

        if not os.path.exists(self.log_img):
            os.mkdir(self.log_img)

        logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                            filename=os.path.join(self.log_dir, 'Trainer_{}.log'.format(
                                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                            )),
                            filemode='w',
                            format=
                            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(logdir=self.log_dir)
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.learn_rate_g = learn_rate * 10
        self.decay_steps = decay_steps
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.adam_optimizer = adam_optimizer
        self.cuda = cuda and torch.cuda.is_available()
        self.init()
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learn_rate_g,
            betas=(self.adam_optimizer['beta1'], self.adam_optimizer['beta2']),
            eps=self.adam_optimizer['epsilon'])

        self.optimizer_Dr = torch.optim.Adam(
            self.discriminatorR.parameters(),
            lr=self.learn_rate,
            betas=(self.adam_optimizer['beta1'], self.adam_optimizer['beta2']),
            eps=self.adam_optimizer['epsilon'])

        self.optimizer_Da = torch.optim.Adam(
            self.discriminatorA.parameters(),
            lr=self.learn_rate,
            betas=(self.adam_optimizer['beta1'], self.adam_optimizer['beta2']),
            eps=self.adam_optimizer['epsilon'])

        # lr decay
        self.scheduler_g = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.decay_steps, gamma=0.9)
        self.scheduler_a = optim.lr_scheduler.StepLR(self.optimizer_Da, step_size=self.decay_steps, gamma=0.9)
        self.scheduler_r = optim.lr_scheduler.StepLR(self.optimizer_Dr, step_size=self.decay_steps, gamma=0.9)

        self.train_dh = DataHandler(train_dir, patch_size=self.patch_size, augment=True)
        self.val_dh = DataHandler(val_dir, patch_size=self.patch_size)

        self.train_loader = Data.DataLoader(
            self.train_dh, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        self.val_loader = Data.DataLoader(
            self.val_dh, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        if self.cuda:
            self.generator.cuda()
            self.discriminatorR.cuda()
            self.discriminatorA.cuda()
            self.adversarial_loss.cuda()
            self.mse_loss.cuda()



    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.learn_rate * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def weights_init_normal(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # Initialize weights
    def init(self):
        self.generator.apply(self.weights_init_normal)
        self.discriminatorR.apply(self.weights_init_normal)
        self.discriminatorA.apply(self.weights_init_normal)

    def train_discriminatorR(self, source_batch, target_batch, np_target_batch):

        real_label = torch.ones(source_batch.shape[0], dtype=torch.float, requires_grad=False)
        fake_label = torch.zeros(source_batch.shape[0], dtype=torch.float, requires_grad=False)
        if self.cuda:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
        # Reset gradients
        self.optimizer_Dr.zero_grad()
        # 1.1 Train on Real Batch
        real_output = self.discriminatorR(target_batch)
        real_loss = self.adversarial_loss(torch.squeeze(real_output), real_label)
        # 1.2 Train on Fake Batch
        gen_target_batch = self.generator(source_batch)
        fake_output = self.discriminatorR(gen_target_batch)
        fake_loss = self.adversarial_loss(torch.squeeze(fake_output), fake_label)
        # 1.3 Train on Real No-Pair Batch
        np_real_output = self.discriminatorR(np_target_batch)
        np_real_loss = self.adversarial_loss(torch.squeeze(np_real_output), real_label)
        dr_loss = (real_loss + fake_loss + np_real_loss) / 3
        dr_loss.backward()
        # Update weights with gradients
        self.optimizer_Dr.step()

        return dr_loss

    def train_discriminatorA(self, source_batch, target_batch, np_target_batch):

        real_label = torch.ones(source_batch.shape[0], dtype=torch.float, requires_grad=False)
        fake_label = torch.zeros(source_batch.shape[0], dtype=torch.float, requires_grad=False)
        if self.cuda:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
        # Reset gradients
        self.optimizer_Da.zero_grad()
        # 1.1 Train on Real Pair
        real_pair = torch.cat((source_batch, target_batch), 1)
        real_output_a = self.discriminatorA(real_pair)
        real_loss_a = self.adversarial_loss(torch.squeeze(real_output_a), real_label)
        # 1.2 Train on Fake Pair
        gen_target_batch = self.generator(source_batch)
        fake_pair = torch.cat((source_batch, gen_target_batch), 1)
        fake_output_a = self.discriminatorA(fake_pair)
        fake_loss_a = self.adversarial_loss(torch.squeeze(fake_output_a), fake_label)
        # 1.3 Train on No-Pair
        np_pair = torch.cat((source_batch, np_target_batch), 1)
        np_output_a = self.discriminatorA(np_pair)
        np_loss_a = self.adversarial_loss(torch.squeeze(np_output_a), fake_label)
        da_loss = (real_loss_a + fake_loss_a + np_loss_a) / 3
        da_loss.backward()
        # Update weights with gradients
        self.optimizer_Da.step()

        return da_loss

    def train_generator(self, source_batch, target_batch):

        self.generator.train()
        real_label = torch.ones(source_batch.shape[0], dtype=torch.float, requires_grad=False)
        if self.cuda:
            real_label = real_label.cuda()

        # Reset gradients
        self.optimizer_G.zero_grad()

        gen_target_batch = self.generator(source_batch)
        # 1.1 Train discriminatorR
        gen_output = self.discriminatorR(gen_target_batch)
        gen_loss_d = self.adversarial_loss(torch.squeeze(gen_output), real_label)
        # 1.2 Train discriminatorA
        gen_pair = torch.cat((source_batch, gen_target_batch), 1)
        gen_output_a = self.discriminatorA(gen_pair)
        gen_loss_a = self.adversarial_loss(torch.squeeze(gen_output_a), real_label)
        # 1.2 Train mse
        mse_loss = self.mse_loss(gen_target_batch, target_batch)

        gen_loss = (gen_loss_d + gen_loss_a) / 2 + mse_loss
        gen_loss.backward()
        # Update weights with gradients
        self.optimizer_G.step()
        return gen_loss

    def lr_decay(self):
        self.scheduler_g.step()
        self.scheduler_r.step()
        self.scheduler_a.step()

    def epoch_n_from_weights_name(self, w_name):
        """
        Extracts the last epoch number from the standardized weights name.
            :discriminatorR_epoch_6.pkl
        """
        try:
            starting_epoch = int(w_name.split('_')[-1].rstrip('.pkl'))
        except Exception as e:
            self.logger.warning(
                'Could not retrieve starting epoch from the weights name: \n{}'.format(w_name)
            )
            self.logger.error(e)
            starting_epoch = 0
        return starting_epoch

    def process_output(self, input):
        return (input + 1.0) / 2.0

    def remove_old_weights(self, max_n_weights=5):
        """
        Scans the weights folder and removes all but:
            - max_n_weights most recent 'others' weights.
        """
        w_list = {}
        w_list['g'] = [w for w in os.scandir(self.weight_dir) if
                       w.name.endswith('.pkl') and w.name.startswith('generator')]
        w_list['dr'] = [w for w in os.scandir(self.weight_dir) if
                        w.name.endswith('.pkl') and w.name.startswith('discriminatorR')]
        w_list['da'] = [w for w in os.scandir(self.weight_dir) if
                        w.name.endswith('.pkl') and w.name.startswith('discriminatorA')]

        epochs_set = {}
        for type in ['g', 'dr', 'da']:
            epochs_set[type] = [self.epoch_n_from_weights_name(w.name) for w in w_list[type]]

            if len(w_list[type]) > max_n_weights:
                epoch_list = np.sort(epochs_set[type])[::-1]
                epoch_list = epoch_list[0:max_n_weights]
                for w in w_list[type]:
                    if self.epoch_n_from_weights_name(w.name) not in epoch_list:
                        os.remove(w.path)

    def save_weights(self, epoch):

        torch.save(self.generator.state_dict(),
                   os.path.join(self.weight_dir, 'generator_epoch_{}.pkl'.format(epoch)))
        torch.save(self.discriminatorA.state_dict(),
                   os.path.join(self.weight_dir, 'discriminatorA_epoch_{}.pkl'.format(epoch)))
        torch.save(self.discriminatorR.state_dict(),
                   os.path.join(self.weight_dir, 'discriminatorR_epoch_{}.pkl'.format(epoch)))

        try:
            self.remove_old_weights(self.max_n_weights)
        except Exception as e:
            self.logger.warning('Could not remove weights: {}'.format(e))

    def validation(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            with tqdm(desc='validation %d' % epoch, total=len(self.val_loader)) as pbar:
                for i, data_batch in enumerate(self.val_loader):
                    source_batch, target_batch, _ = data_batch
                    if self.cuda:
                        source_batch = source_batch.cuda()
                        target_batch = target_batch.cuda()

                    gen_batch = self.generator(source_batch)

                    gen_batch = self.process_output(gen_batch)  # [-1, 1]->[0, 1]
                    source_batch = self.process_output(source_batch)
                    target_batch = self.process_output(target_batch)

                    concat = torch.cat([source_batch, target_batch, gen_batch], 3)
                    x = vutils.make_grid(concat, normalize=True, scale_each=True)
                    self.writer.add_image('Image/%d' % i, x, epoch)

                    save_image(gen_batch, os.path.join(self.log_img, "{}_{}_gen.png".format(epoch, i)), scale_each=True)
                    save_image(source_batch, os.path.join(self.log_img, "{}_{}_src.png".format(epoch, i)), scale_each=True)
                    save_image(target_batch, os.path.join(self.log_img, "{}_{}_tar.png".format(epoch, i)), scale_each=True)
                    save_image(x, os.path.join(self.log_img, "{}_{}.png".format(epoch, i)), scale_each=True)

                    pbar.update()

    def train(self, epochs):
        for epoch in range(epochs):
            with tqdm(desc='epoch %d' % epoch, total=len(self.train_loader)) as pbar:
                for i, data_batch in enumerate(self.train_loader):
                    source_batch, target_batch, np_target_batch = data_batch
                    # 获取输入
                    if self.cuda:
                        source_batch = source_batch.cuda()
                        target_batch = target_batch.cuda()
                        np_target_batch = np_target_batch.cuda()

                    dr_loss = self.train_discriminatorR(source_batch, target_batch, np_target_batch)
                    da_loss = self.train_discriminatorA(source_batch, target_batch, np_target_batch)
                    if i % 5 == 0:
                        g_loss = self.train_generator(source_batch, target_batch)

                        self.logger.debug(
                            "[Epoch %d/%d] [Batch %d/%d] [Dr loss: %f] [Da loss: %f] [G loss: %f]"
                            % (epoch, epochs, i, len(self.train_loader), dr_loss.item(), da_loss.item(), g_loss.item())
                        )

                    if i % 100 == 0:
                        self.writer.add_scalar('Loss/G_loss', g_loss.item(), epoch * len(self.train_loader) + i)
                        self.writer.add_scalar('Loss/Dr_loss', dr_loss.item(), epoch * len(self.train_loader) + i)
                        self.writer.add_scalar('Loss/Da_loss', da_loss.item(), epoch * len(self.train_loader) + i)
                        for param_group in self.optimizer_G.param_groups:
                            self.writer.add_scalar('LR', param_group['lr'], epoch * len(self.train_loader) + i)

                    pbar.update()

            # val
            if epoch % 10 == 0:
                self.validation(epoch)
            # save
            self.save_weights(epoch)
            # lr decay
            self.lr_decay()














