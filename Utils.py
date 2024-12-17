import torch
import torchvision
import torchvision.transforms as transforms
from randaug import RandAugment

def get_loaders_CIFAR10(size, batch_size, aug=True,N = 2, M = 14):
  transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.Resize(size),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

  transform_test = transforms.Compose([
  transforms.Resize(size),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  # Add RandAugment with N, M(hyperparameter)
  if aug:
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))
  # Prepare dataset
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
  download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
  shuffle=True, num_workers=8)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
  download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
  num_workers=8)
  return trainloader, testloader

def get_loaders_CIFAR100(size, batch_size, aug=True,N = 2, M = 14):
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  
  transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  
  # Add RandAugment with N, M(hyperparameter)
  if aug:
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))
  # Prepare dataset
  trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
  download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
  shuffle=True, num_workers=8)
  testset = torchvision.datasets.CIFAR100(root='./data', train=False,
  download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
  num_workers=8)
  return trainloader, testloader