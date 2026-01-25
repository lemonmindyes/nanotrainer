import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class MNISTDataset(Dataset):

    def __init__(self, root = './data', train = True, download = True, transform = None):
        super().__init__()
        self.mnist = MNIST(root, train, download = download)
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image = self.mnist[idx][0]
        image = self.transform(image)
        label = self.mnist[idx][1]
        return image, label


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.loss_func(x, y)


def train_step(model, loss_func, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    model.train()
    out = model(x)
    loss = loss_func(out, y)
    return loss


if __name__ == "__main__":
    from nanotrainer import trainer
    from nanotrainer import checkpoint, logging
    from nanotrainer import lr_scheduler, single

    device = torch.device('cpu')
    max_steps = 10000
    warmup_ratio = 0.05
    gradient_accumulation_steps = 4

    train_dataset = MNISTDataset()
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    model = Net()
    loss_func = Loss()
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    cosine_lr_scheduler = lr_scheduler.CosineWarmupDecay(opt,
                                                         warmup_ratio = warmup_ratio,
                                                         min_lr = 1e-5
                                                         )
    strategy = single.SingleStrategy(
        model,
        opt,
        lr_scheduler = cosine_lr_scheduler,
        gradient_accumulation_steps = gradient_accumulation_steps
    )
    callback = [
        logging.ModelSummaryCallback(),
        logging.LoggingCallback(log_interval = 300),
        logging.RealtimePlotCallback(max_points = 1000, log_interval = 50),
        checkpoint.CheckpointCallback('./model/mnist',
                                      save_interval = 1500,
                                      )
    ]

    trainer = trainer.Trainer(
        device = device,
        max_steps = max_steps,

        model = model,
        loss_func = loss_func,
        optimizer = opt,
        dataloader = train_loader,
        train_step = train_step,
        strategy = strategy,
        callback = callback
    )
    trainer.fit()














