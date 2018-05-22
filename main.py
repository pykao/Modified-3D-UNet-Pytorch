import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import load_dataset, BraTS2018List
from model import Modified3DUNet
import paths

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

#print datestr()



# Training setting
parser = argparse.ArgumentParser(description='PyTorch Modified 3D U-Net Training')
#parser.add_argument('-m', '--modality', default='T1', choices = ['T1', 'T1c', 'T2', 'FLAIR'],
#                    type = str, help='modality of input 3d images (default:T1)')
#parser.add_argument('-w', '--workers', default=8, type=int,
#                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    help='batch size (default: 2)')
parser.add_argument('-g', '--gpu', default='0', type=str)
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    help='initial learning rate (default:5e-4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=985e-3, type=float,
                     help='weight decay (default: 985e-3)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('-d', '--data', default=paths.preprocessed_training_data_folder,
                    type=str, help='The location of BRATS2015')

log_file = os.path.join("train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


global args, best_loss
best_loss = float('inf')
args = parser.parse_args()
#print os.environ['CUDA_VISIBLE_DEVICES']
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# input = data.to(device)

# Loading the model
in_channels = 4
n_classes = 4
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
#print args.data


# Split the training and testing dataset

test_size = 0.1
train_idx, test_idx = train_test_split(range(285), test_size = test_size)
train_data = load_dataset(train_idx)
test_data = load_dataset(test_idx)


#print all_data.keys()
# create your optimizer
#optimizer = optim.adam(net.parameteres(), lr=)
'''
# in training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step
'''
