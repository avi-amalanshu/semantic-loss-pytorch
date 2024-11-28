import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
from lightning.pytorch import loggers
import numpy as np
from semantic_loss_pytorch import SemanticLoss


class SemiSupervisedMNIST(pl.LightningModule):
    def __init__(self, keep_prob=0.5, semantic_weight=5e-4, learning_rate=1e-4):
        super().__init__()
        self.extract = nn.Sequential(nn.Linear(784, 1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 500),
                                     nn.ReLU(),
                                     nn.Linear(500, 250),
                                     nn.ReLU(),
                                     nn.Linear(250, 250),
                                     nn.ReLU(),
                                     nn.Linear(250, 250),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(250, eps=0.001, momentum=0.99),
                                     nn.Dropout(p=1-keep_prob))
        self.classify = nn.Linear(250, 10)
        self.semantic_weight = semantic_weight
        self.learning_rate = learning_rate

        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.)
        self.extract.apply(init_weights)
        torch.nn.init.xavier_uniform_(self.classify.weight)
        self.classify.bias.data.fill_(0.)

        self.xentropy = nn.BCEWithLogitsLoss(reduction='none')
        # self.sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
        # self.crop = torchvision.transforms.v2.Compose([torchvision.transforms.v2.RandomCrop(size=25),
        #                                                torchvision.transforms.v2.Resize(size=26, antialias=True),
        #                                                torchvision.transforms.v2.Pad(padding=1)])
        self.crop = torchvision.transforms.v2.Compose([torchvision.transforms.v2.RandomCrop(size=25),
                                                       torchvision.transforms.v2.Pad(padding=(1,1,2,2))])
    def standardize(self, image):
        """
        Matches tf.image.per_image_standardization ops:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py
        """
        # assuming [N, C, H, W] tensors by convention
        num_pixels = 28 * 28
        if image.dim() < 3:
            raise ValueError("Input image must have at least 3 dimensions.")
        image = image.to(torch.float32)
        image_mean = image.mean(dim=(-1, -2), keepdim=True)
        image_stddev = image.std(dim=(-1, -2), keepdim=True)
        # protect against small vars
        min_stddev = torch.rsqrt(torch.tensor(float(num_pixels), dtype=torch.float32))
        adjusted_stddev = torch.maximum(image_stddev, min_stddev)

        standardized_image = (image - image_mean) / adjusted_stddev

        return standardized_image

    def augment(self, x):

        # Add Gaussian noise
        noise = torch.randn_like(x) * 0.3
        x = x + noise
        # Random crop to 25x25 and pad to 28x28
        x = self.crop(x)
        return x

    def forward(self, x):
        x = self.standardize(x)

        if self.training is True:
            x = self.augment(x)

        x = x.view(-1, 784)
        x = self.extract(x)
        x = self.classify(x)
        return x

    def compute_wmc(self, input):
        normalized_logits = torch.sigmoid(input)
        # normalized_logits = torch.softmax(input, dim=-1)
        # import pdb; pdb.set_trace()
        batch_size = input.size(0)
        wmc = torch.zeros(batch_size, device=input.device)

        for i in range(10):
            one_situation = torch.ones(batch_size, 10, device=input.device)
            one_situation[:, i] = 0
            wmc += torch.prod(one_situation - normalized_logits, dim=1)

        return torch.mean(torch.abs(wmc))

    def compute_wmc_2(self, input):
        normalized_logits = torch.sigmoid(input)
        # normalized_logits = torch.softmax(input, dim=-1).unsqueeze(1).repeat(1, 10, 1)
        state_probs_in_that_world = torch.where(torch.eye(10, dtype=bool, device=self.device).unsqueeze(0).repeat(len(normalized_logits), 1, 1),
                                                normalized_logits,
                                                1 - normalized_logits)
        world_probs = torch.prod(state_probs_in_that_world, dim=-1)
        wmc_values = torch.sum(world_probs, dim=-1)
        return wmc_values

    def training_step(self, batch, batch_idx):
        images, labels, is_labeled = batch
        labels_onehot = F.one_hot(labels, 10).float()

        # Forward pass with training augmentations
        self.train()
        scores = self(images)

        # Calculate WMC loss
        wmc_values = self.compute_wmc_2(scores)
        log_wmc = torch.log(wmc_values + 1e-10)
        wmc_loss = -self.semantic_weight * log_wmc

        # Calculate cross entropy
        cross_entropy = self.xentropy(scores, labels_onehot).sum(axis=1)
        # semantic_loss = self.sl(probabilities=torch.softmax(scores, dim=-1))

        # Combine losses based on whether examples are labeled
        loss = torch.where(
            is_labeled,
            wmc_loss + cross_entropy,  # labeled examples: both losses
            wmc_loss  # unlabeled examples: only WMC loss
        ).mean()

        # loss = torch.where(
        #     is_labeled,
        #     semantic_loss + cross_entropy,  # labeled examples: both losses
        #     semantic_loss  # unlabeled examples: only WMC loss
        # ).mean()

        # Training acc
        pred = scores.argmax(dim=1)
        # accuracy = (pred == labels)[is_labeled].float().mean() if is_labeled.any() else torch.tensor(0.0)
        accuracy = (pred == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', accuracy)
        # self.log('train_wmc', wmc_values.mean())

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        scores = self(images)
        pred = scores.argmax(dim=1)
        accuracy = (pred == labels).float().mean()
        self.log('val_acc', accuracy)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class SemiSupervisedMNISTDataset(Dataset):
    def __init__(self, dataset, num_labeled=None):
        self.dataset = dataset
        self.num_labeled = num_labeled

        if num_labeled is not None:
            # Get balanced labeled indices
            all_labels = torch.tensor([label for _, label in dataset])
            n_classes = 10
            n_labels_per_class = num_labeled // n_classes

            labeled_indices = []
            for c in range(n_classes):
                class_indices = (all_labels == c).nonzero().view(-1)
                perm = torch.randperm(len(class_indices))
                labeled_indices.extend(class_indices[perm[:n_labels_per_class]].tolist())

            self.labeled_mask = torch.zeros(len(dataset), dtype=torch.bool)
            self.labeled_mask[labeled_indices] = True
        else:
            self.labeled_mask = torch.ones(len(dataset), dtype=torch.bool)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        is_labeled = self.labeled_mask[idx]
        return image, label, is_labeled


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, num_labeled, batch_size):
        super().__init__()
        self.num_labeled = num_labeled
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        datasets.MNIST('./data', train=True, download=True)
        datasets.MNIST('./data', train=False, download=True)

    def setup(self, stage=None):
        full_train = datasets.MNIST('./data', train=True, transform=self.transform)

        if self.num_labeled < len(full_train):
            # indices = torch.randperm(len(full_train))
            # labeled_indices = indices[:self.num_labeled]
            # self.train_dataset = SemiSupervisedMNISTDataset(full_train, labeled_indices)
            self.train_dataset = SemiSupervisedMNISTDataset(full_train, self.num_labeled)
        else:
            self.train_dataset = SemiSupervisedMNISTDataset(full_train)

        self.val_dataset = datasets.MNIST('./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=7,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=7,
                          persistent_workers=True)


def main(args):
    logger = loggers.TensorBoardLogger(save_dir='lightning_logs',
                                       name=f'labeled-{args.num_labeled}_'
                                            f'batch-{args.batch_size}_'
                                            f'sw_{args.semantic_weight}'
                                            f'lr_{args.learning_rate}')
    model = SemiSupervisedMNIST(semantic_weight=args.semantic_weight, learning_rate=args.learning_rate)
    data_module = MNISTDataModule(args.num_labeled, args.batch_size)

    trainer = pl.Trainer(
        max_epochs=10,
        # max_steps=50000,
        val_check_interval=2000,
        log_every_n_steps=2000,
        accelerator='auto',
        # num_nodes=8,
        logger=logger
    )

    trainer.fit(model, data_module)

# count = 0
counts = [0 for _ in range(10)]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labeled', type=int, required=True,
                        help='Number of labeled examples for semi-supervised learning.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for mini-batch Adam gradient descent.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--semantic_weight', type=float, default=5e-4,
                        help='Weight given to the semantic loss term within the total loss computation.')
    args = parser.parse_args()

    main(args)
