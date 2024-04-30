import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from Resnet import ResidualBlock, ResNet18
from training import train
from testing import test

if __name__ == '__main__':
    # 資料預處理
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    # 分割training和valid的資料
    train_dataset, valid_dataset = torch.utils.data.random_split(
        torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        , lengths=[40000, 10000])

    #拿取load的資料
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    vaildloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=1)
    test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=1)

    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    # 建立model
    model = ResNet18(ResidualBlock)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 損失函數、優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 開始訓練
    train(model, trainloader, vaildloader, device, criterion, optimizer)

    # 把訓練好的模型，用test data看看結果精不精確
    test(model, testloader, device, classes)