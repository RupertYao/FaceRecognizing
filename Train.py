import os

import torch
import torch.cuda
from torch import nn, optim

import config
from FaceRecNet import FaceRecNet
from Preprocess import getData

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
PEOPLE_NUM = len(os.listdir(config.DATA_TRAIN))


def trainModel():
    net = FaceRecNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    for epoch in range(config.EPOCHS):
        train_loader, test_loader = getData(batch_size=config.BATCH_SIZE)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            # print("x", x)
            output = net(x)
            # print(output)
            # print(output.max(1))
            # y_one_hot = nn.functional.one_hot(y, y.size)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, (step + 1) * len(x), len(train_loader.dataset),
                100. * (step + 1) / len(train_loader), loss.item()))
    train_loader, test_loader = getData(batch_size=config.BATCH_SIZE)
    test(net, test_loader)
    torch.save(
        net.state_dict(),
        os.path.join(config.DATA_MODEL, config.DEFAULT_MODEL)
        )
    return net


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.NLLLoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            # print("-----output-----")
            # print(output)
            # print("-------y--------")
            # print(y)
            # print('\n')
            test_loss += loss_fn(output, y)
            pred = output.max(1)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        '\ntest loss={:.4f}, accuracy={:.4f}\n'
        .format(test_loss, float(correct) / len(test_loader.dataset))
        )


if __name__ == "__main__":
    trainModel()
