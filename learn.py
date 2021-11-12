import random
import torch
import torch.nn as nn
from dataset import ImageFilesDataset
from torch.utils.data import DataLoader
from model import Segment

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
random.seed(27)
torch.manual_seed(27)
if cuda: torch.cuda.manual_seed(27)


def train(train_dir, batch_size, lr):
    # Instantiate model, data loaders, loss function, and optimizer
    train_set = ImageFilesDataset(train_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    model = Segment()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            x, y = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = model(x)
            y_hat = torch.argmax(y_hat)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


if __name__ == '__main__':
    train('./data', 1, 2)
