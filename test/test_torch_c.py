# -*- coding: UTF-8 -*-
# @Time    : 2022/5/1 14:36
# @File    : test_torch_c.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
# import torch
# import torch.nn as nn
# import antgo.net.torch2paddle.nn as nn
import antgo.framework.torch2paddle.mapper as torch
import antgo.framework.torch2paddle.mapper.nn as nn
import antgo.framework.torch2paddle.mapper.optimizer as optim
import time
import os
from MobileNetV2 import *
import argparse
from read_ImageNetData import ImageNetData



def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
  since = time.time()
  resumed = False

  best_model_wts = model.state_dict()

  for epoch in range(args.start_epoch + 1, num_epochs):

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        if args.start_epoch > 0 and (not resumed):
          scheduler.step(args.start_epoch + 1)
          resumed = True
        else:
          scheduler.step(epoch)
        model.train(True)  # Set model to training mode
      else:
        model.train(False)  # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      tic_batch = time.time()
      # Iterate over data.
      for i, (inputs, labels) in enumerate(dataloders[phase]):
        # wrap them in Variable
        # if use_gpu:
        #   inputs = Variable(inputs.cuda())
        #   labels = Variable(labels.cuda())
        # else:
        #   inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

        batch_loss = running_loss / ((i + 1) * args.batch_size)
        batch_acc = running_corrects / ((i + 1) * args.batch_size)

        if phase == 'train' and i % args.print_freq == 0:
          print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
            epoch, num_epochs - 1, i, round(dataset_sizes[phase] / args.batch_size) - 1, scheduler.get_lr()[0], phase,
            batch_loss, batch_acc.item(), \
                   args.print_freq / (time.time() - tic_batch)))
          tic_batch = time.time()

        break

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc.item()))

      break

    if (epoch + 1) % args.save_epoch_freq == 0:
      if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
      torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

    break

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
  parser.add_argument('--data-dir', type=str, default="/ImageNet")
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--num-class', type=int, default=1000)
  parser.add_argument('--num-epochs', type=int, default=100)
  parser.add_argument('--lr', type=float, default=0.045)
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--gpus', type=str, default=0)
  parser.add_argument('--print-freq', type=int, default=10)
  parser.add_argument('--save-epoch-freq', type=int, default=1)
  parser.add_argument('--save-path', type=str, default="output")
  parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
  parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
  args = parser.parse_args()

  # read data
  dataloders, dataset_sizes = ImageNetData(args)

  # use gpu or not
  use_gpu = torch.cuda.is_available()
  print("use_gpu:{}".format(use_gpu))

  # get model
  model = mobilenetv2_19(num_classes=args.num_class)

  if args.resume:
    if os.path.isfile(args.resume):
      print(("=> loading checkpoint '{}'".format(args.resume)))
      checkpoint = torch.load(args.resume)
      base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
      model.load_state_dict(base_dict)
    else:
      print(("=> no checkpoint found at '{}'".format(args.resume)))

  if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

  # define loss function
  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

  model = train_model(args=args,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer_ft,
                      scheduler=exp_lr_scheduler,
                      num_epochs=args.num_epochs,
                      dataset_sizes=dataset_sizes)
