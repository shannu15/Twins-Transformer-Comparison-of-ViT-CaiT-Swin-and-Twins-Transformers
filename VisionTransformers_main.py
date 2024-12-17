import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import csv
import time
from randaug import RandAugment
from models.cait import CaiT
print(sys.path)
import Utils
import models.twins as twins

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='CIFAR10') # options: CIFAR10, CIFAR100
parser.add_argument('--dataset_classes', default='10') # options: 10 for CIFAR10,
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') #
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--net', default='twins') # options: vit, swin, cait, twins
parser.add_argument('--heads', default='6')
parser.add_argument('--layers', default='12') # depth
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='64') # was 512
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="420", type=int) # or 512
args = parser.parse_args()
best_acc = 0

def count_parameters(model): # count number of trainable parameters in the model
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
  bs = int(args.bs)
  imsize = int(args.size)
  use_amp = not args.noamp
  aug = args.noaug

  best_acc = 0 # best test accuracy
  start_epoch = 0 # start from epoch 0 or last checkpoint epoch
  # Data
  print('==> Preparing data..')
  size = imsize
  if args.dataset == "CIFAR10":
    trainloader, testloader = Utils.get_loaders_CIFAR10(size, bs)
  if args.dataset == "CIFAR100":
    trainloader, testloader = Utils.get_loaders_CIFAR100(size, bs)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  # Model factory..
  if args.net=="cait":
    net = CaiT(
      image_size = size,
      patch_size = args.patch,
      num_classes = int(args.dataset_classes),
      dim = int(args.dimhead),
      depth = int(args.layers), # depth of transformer for patch to patch attention only
      cls_depth=2, # depth of cross attention of CLS tokens to patch
      heads = int(args.heads),
      mlp_dim = int(args.dimhead)*4, #512,
      dropout = 0.1,
      emb_dropout = 0.1,
      layer_dropout = 0.05
      )
  elif args.net == "twins":
    net = twins.Twins(
      num_classes= int(args.dataset_classes),
      img_size=size, patch_size=args.patch,
      embed_dims=[64, 128, 320, 512],
      num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
      depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
      drop_rate=0.0,
      drop_path_rate=0.1,)
  
  pcount = count_parameters(net)
  print("count of parameters in the model = ", pcount/1e6, " million")

  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
  
  # Loss is CE
  criterion = nn.CrossEntropyLoss()
  if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
  elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

  # use cosine scheduling
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
  ##### Training
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
  list_loss = []
  list_acc = []

  net.cuda()
  for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch, net, trainloader, criterion,scaler,optimizer, use_amp)
    val_loss, acc = test(epoch, net, testloader, criterion, optimizer, scaler)
    scheduler.step(epoch-1) # step cosine scheduling
    list_loss.append(val_loss)
    list_acc.append(acc)
  
def train(epoch, net, trainloader, criterion, scaler, optimizer, use_amp):
  print('\nEpoch: %d' % epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    # Train with amp
    with torch.cuda.amp.autocast(enabled=use_amp):
      outputs = net(inputs)
      loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
  print(batch_idx, len(trainloader), 'Loss: %.3f | TrainAcc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
  return train_loss/(batch_idx+1)
  
##### Validation
# for other than LLP model, use following for test
def test(epoch, net, testloader, criterion, optimizer, scaler):
  global best_acc
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)
      test_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
    
    print(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
  
  # Save checkpoint.
  acc = 100.*correct/total
  if acc > best_acc:
    print('Saving..')
    state = {"model": net.state_dict(),
      "optimizer": optimizer.state_dict(),
      "acc": acc,
      "epoch": epoch,
      "scaler": scaler.state_dict()}
    
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
    best_acc = acc
  return test_loss, acc

if __name__ == "__main__":
  sys.exit(int(main() or 0))