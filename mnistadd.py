import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from itertools import product
from typing import Tuple, List, Optional


class MNISTAddDataset(Dataset):
    """
    Row-major indexing for pairs of MNIST images.
    Indices (4, 8493) from MNIST trainset -> 4*60000 + 8493 = idx 248493 in this dataset's trainset.
    Index 489339 from this dataset's testset -> indices ((489339 // 10000), 489339 % 10000) = (48, 9339) in MNIST testset
    """
    def __init__(self, root: str, train: bool = True):
        # Load MNIST dataset
        self.train = train
        self.mnist = datasets.MNIST(
            root=root,
            train=self.train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    def __len__(self):
        return 60000 ** 2 if self.train else 10000 ** 2

    def __getitem__(self, idx: int):
        # Get two random digits
        idx1, idx2 = idx // len(self), idx % len(self)
        img1, label1 = self.mnist[idx1]
        img2, label2 = self.mnist[idx2]

        # Combine images side by side
        combined_img = torch.cat([img1, img2], dim=2)

        # Calculate sum label and return individual digits
        y = label1 + label2
        c = torch.tensor([label1, label2])

        return combined_img, c, y


class DigitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 possible digits

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class WorldsQueriesMatrix:
    def __init__(self, n_digits: int = 10):
        self.n_digits = n_digits
        self.matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        possible_worlds = list(product(range(self.n_digits), repeat=2))
        n_worlds = len(possible_worlds)
        n_queries = 2 * self.n_digits - 1  # Possible sums from 0 to 18

        w_q = torch.zeros(n_worlds, n_queries)

        for w, (digit1, digit2) in enumerate(possible_worlds):
            sum_value = digit1 + digit2
            w_q[w, sum_value] = 1

        return w_q


class SemanticLoss(nn.Module):
    def __init__(self, n_digits: int = 10):
        super().__init__()
        self.worlds_queries = WorldsQueriesMatrix(n_digits)

    def forward(self, digit_probs: torch.Tensor, sum_labels: torch.Tensor) -> torch.Tensor:
        # Split probabilities for each digit
        prob_digit1, prob_digit2 = digit_probs[:, 0], digit_probs[:, 1]

        # Compute worlds probability
        Z_1 = prob_digit1.unsqueeze(-1)  # [B, 10, 1]
        Z_2 = prob_digit2.unsqueeze(1)  # [B, 1, 10]
        worlds_prob = (Z_1 * Z_2).view(-1, 100)  # [B, 100]

        # Compute query probabilities
        query_prob = torch.matmul(worlds_prob, self.worlds_queries.matrix.to(worlds_prob.device))

        # Add small offset and normalize
        query_prob = query_prob + 1e-5
        query_prob = query_prob / query_prob.sum(dim=-1, keepdim=True)

        # Compute negative log likelihood
        loss = F.nll_loss(query_prob.log(), sum_labels)
        return loss


class MNISTAddModel(pl.LightningModule):
    def __init__(self,
                 learning_rate: float = 1e-3,
                 semantic_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()

        # Model components
        self.encoder = DigitEncoder()
        self.semantic_loss = SemanticLoss()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.semantic_weight = semantic_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the combined image into two halves
        x1, x2 = torch.split(x, x.size(-1) // 2, dim=-1)

        # Get digit probabilities
        logits1 = self.encoder(x1)
        logits2 = self.encoder(x2)

        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)

        return torch.stack([probs1, probs2], dim=1)

    def training_step(self, batch, batch_idx):
        x, sum_labels, digit_labels = batch
        digit_probs = self(x)

        # Compute semantic loss
        sem_loss = self.semantic_loss(digit_probs, sum_labels)

        # Compute classification loss for individual digits
        digit_logits = digit_probs.log()
        class_loss = F.nll_loss(digit_logits.view(-1, 10), digit_labels.view(-1))

        # Combined loss
        loss = class_loss + self.semantic_weight * sem_loss

        self.log('train_loss', loss)
        self.log('train_semantic_loss', sem_loss)
        self.log('train_class_loss', class_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, sum_labels, digit_labels = batch
        digit_probs = self(x)

        # Compute semantic loss
        sem_loss = self.semantic_loss(digit_probs, sum_labels)

        # Compute classification loss
        digit_logits = digit_probs.log()
        class_loss = F.nll_loss(digit_logits.view(-1, 10), digit_labels.view(-1))

        # Combined loss
        loss = class_loss + self.semantic_weight * sem_loss

        # Compute accuracy
        pred_digits = digit_probs.argmax(dim=-1)
        acc = (pred_digits == digit_labels).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_mnist_add():
    # Initialize data
    train_dataset = MNISTAddDataset(root='./data', train=True)
    val_dataset = MNISTAddDataset(root='./data', train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # Initialize model
    model = MNISTAddModel(learning_rate=1e-3, semantic_weight=1.0)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        logger=pl.loggers.TensorBoardLogger('mnist_add_logs/', name='mnist_add')
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train_mnist_add()