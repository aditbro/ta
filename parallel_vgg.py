import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import random
import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.autograd import Variable
from torchvision.models import vgg16
from torch.multiprocessing import Process
from torchvision import datasets, transforms

class CIFAR10Dataset(torch.utils.data.Dataset):
    total_length = 50000
    
    def __init__(self, start_idx, end_idx):
        self.current_batch = 0
        self.batch = {}
        self.max_batch = 5
        self.start_idx = start_idx
        self.end_idx = end_idx
        
    def __len__(self):
        return int(self.end_idx - self.start_idx)
    
    def transform(self, img):
        img = self.toTensor(img)
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        
        return img
        
    def toTensor(self, img):
        img = np.interp(img, (0, 255), (0, 1))
        img = torch.Tensor(img)
        img = img.type(torch.FloatTensor)
        return img
    
    def __getitem__(self, idx):
        idx = idx + self.start_idx
        batch = (idx / 10000) + 1
        
        if batch != self.batch:
            self.load_batch(batch)
            
        idx = idx % 10000
        img = self.batch[idx][0]
        img = self.transform(img)
        return (img, int(self.batch[idx][1]))
    
    def load_batch(self, batch_num):
        file = '../cifar10-python/data_batch_' + str(int(batch_num))
        with open(file, 'rb') as fo:
            self.batch = pickle.load(fo, encoding='bytes')
        
        self.format_batch()
        
    def format_batch(self):
        data = self.batch[b'data']
        labels = self.batch[b'labels']
        self.batch = []
        
        for i in range(len(data)):
            self.batch.append([
                data[i].reshape(3, 32, 32),
                labels[i]
            ])

def get_dataloader(batch_size):
    """ Partitioning MNIST """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    partition_size = int(CIFAR10Dataset.total_length / world_size)

    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    dataset = CIFAR10Dataset(start_idx, end_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)
    return train_loader

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    total_gradient_data = 0
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        total_gradient_data += param.grad.data.nelement()

    print('total tensor size {}'.format(total_gradient_data))

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    batch_size = 100
    torch.manual_seed(1234)
    dataloader = get_dataloader(batch_size)
    model = vgg16()
    # model = model.cuda(0)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()

    print('starting training from ' + str(dist.get_rank()))
    for epoch in range(10):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            start_time = time.time()

            data, target = data
            # data = data.cuda(0)
            # target = target.cuda(0)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            epoch_loss += loss
            loss.backward()

            gradient_time = time.time()
            average_gradients(model)
            gradient_time = time.time() - gradient_time
            print('averaging gradient time {}'.format(gradient_time))

            optimizer.step()

            elapsed_time = time.time() - start_time
            print('rank {} epoch_loss {} time {}'.format(rank, epoch_loss, elapsed_time))
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss)

def init_process():
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '167.205.32.100'
    os.environ['MASTER_PORT'] = '29500'
    rank = int(os.environ['RANK'])
    size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group('gloo', rank=rank, world_size=size)
    print('process initiated rank {}'.format(rank))
    run(rank, size)

if __name__ == '__main__':
    init_process()