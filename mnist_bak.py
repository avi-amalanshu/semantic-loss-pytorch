import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np


class SemiSupervisedMNIST(pl.LightningModule):
    def __init__(self, semantic_weight=0.0005, keep_prob=0.5):
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
                                     nn.BatchNorm1d(250),
                                     nn.Dropout(p=1-keep_prob))
        self.classify = nn.Linear(250, 10)
        self.xentropy = nn.BCEWithLogitsLoss()
        self.semantic_weight = semantic_weight

    def standardize(self, image):
        """
        Matches tf.image.per_image_standardization exactly:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py
        """
        image = image.view(-1, 28, 28, 1)
        num_pixels = 28 * 28
        if image.dim() < 3:
            raise ValueError("Input image must have at least 3 dimensions.")

        # Convert the image to float32 if it's not already
        image = image.to(torch.float32)

        # Compute the mean over the last 3 dimensions (height, width, channels)
        image_mean = image.mean(dim=(-1, -2, -3), keepdim=True)

        # Compute the standard deviation over the last 3 dimensions (height, width, channels)
        image_stddev = image.std(dim=(-1, -2, -3), keepdim=True)

        # Compute the minimum standard deviation to protect against small variances
        min_stddev = torch.rsqrt(torch.tensor(float(num_pixels), dtype=torch.float32))

        # Adjust the standard deviation by taking the maximum of the computed stddev and min_stddev
        adjusted_stddev = torch.maximum(image_stddev, min_stddev)

        # Subtract the mean and divide by the adjusted standard deviation
        standardized_image = (image - image_mean) / adjusted_stddev

        return standardized_image

    def augment(self, x):
        batch_size = x.size(0)

        # Add Gaussian noise
        noise = torch.randn_like(x) * 0.3
        x = x + noise

        # Random crop and resize
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Crop to 25x25
        crop_size = 25
        margin = (28 - crop_size) // 2
        start_h = torch.randint(0, margin * 2 + 1, (batch_size,))
        start_w = torch.randint(0, margin * 2 + 1, (batch_size,))

        cropped = []
        for i in range(batch_size):
            h, w = start_h[i], start_w[i]
            crop = x[i:i + 1, :, h:h + crop_size, w:w + crop_size]
            # Resize back to 28x28
            crop = F.interpolate(crop, size=(28, 28), mode='bilinear', align_corners=False)
            cropped.append(crop)

        x = torch.cat(cropped, dim=0)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
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

        # Initialize wmc_tmp as a tensor of zeros
        batch_number = input.shape[0]
        wmc_tmp = torch.zeros(batch_number, device=normalized_logits.device)

        # Loop to calculate WMC
        for i in range(10):
            one_situation = torch.cat([
                torch.cat([torch.ones(batch_number, i, device=normalized_logits.device),
                           torch.zeros(batch_number, 1, device=normalized_logits.device)], dim=1),
                torch.ones(batch_number, 10 - i - 1, device=normalized_logits.device)
            ], dim=1)

            # Calculate the product for each situation and add it to wmc_tmp
            wmc_tmp += torch.prod(one_situation - normalized_logits, dim=1)

        wmc_tmp = torch.abs(wmc_tmp)
        wmc = torch.mean(wmc_tmp)
        return wmc

    def training_step(self, batch, batch_idx):
        images, labels, is_labeled = batch
        labels_onehot = F.one_hot(labels, 10).float()

        # Forward pass with training augmentations
        self.train()
        scores = self(images)

        # Calculate WMC loss for all examples
        wmc_values = self.compute_wmc(scores)
        log_wmc = torch.log(wmc_values + 1e-10)
        wmc_loss = -self.semantic_weight * log_wmc

        # Calculate cross entropy only for labeled examples
        cross_entropy = self.xentropy(scores, labels_onehot)

        # Combine losses based on whether examples are labeled
        # count = sum(is_labeled)
        # if count != 0:
        #     print(f'step {batch_idx}: is_labeled = \n{is_labeled}\nCount = {sum(is_labeled)}')
        loss = torch.where(
            is_labeled,
            wmc_loss + cross_entropy,  # labeled examples: both losses
            wmc_loss  # unlabeled examples: only WMC loss
        ).mean()

        # Calculate accuracy only for labeled examples
        pred = scores.argmax(dim=1)
        # accuracy = (pred == labels)[is_labeled].float().mean() if is_labeled.any() else torch.tensor(0.0)
        accuracy = (pred == labels).float().mean()

        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', accuracy)
        self.log('train_wmc', wmc_values.mean())
        # self.log('num_labeled', count)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        scores = self(images)
        pred = scores.argmax(dim=1)
        accuracy = (pred == labels).float().mean()
        self.log('val_acc', accuracy)
        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class SemiSupervisedMNISTDataset(Dataset):
    def __init__(self, dataset, labeled_indices=None):
        self.dataset = dataset
        if labeled_indices is not None:
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
            indices = torch.randperm(len(full_train))
            labeled_indices = indices[:self.num_labeled]
            self.train_dataset = SemiSupervisedMNISTDataset(full_train, labeled_indices)
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
    model = SemiSupervisedMNIST(semantic_weight=args.semantic_weight)
    data_module = MNISTDataModule(args.num_labeled, args.batch_size)

    trainer = pl.Trainer(
        max_epochs=20,
        # max_steps=50000,
        val_check_interval=500,
        log_every_n_steps=200,
        accelerator='auto',
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labeled', type=int, required=True,
                        help='Number of labeled examples for semi-supervised learning')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for mini-batch Adam gradient descent')
    parser.add_argument('--semantic_weight', type=float, required=True, default=0.0005,
                        help='Semantic Weight')
    args = parser.parse_args()

    main(args)
