from trainer import Trainer
from tester import Tester
from model import Generator, DiscriminatorA, DiscriminatorR


def train():
        generator = Generator()
        discriminatorA = DiscriminatorA()
        discriminatorR = DiscriminatorR()

        trainer = Trainer(generator=generator,
                discriminatorR=discriminatorR,
                discriminatorA=discriminatorA,
                train_dir='./data/test/',
                val_dir='./data/test/',
                batch_size=2,
                patch_size=64,
                learn_rate=0.0002,
                cuda=True)

        trainer.train(epochs=1000)

def test():
        generator = Generator()
        tester = Tester(generator, batch_size=32, cuda=False, self_dir='./data/test/', self_test=True)
        tester.test()

#train()
test()