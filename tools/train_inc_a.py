# coding=utf-8
import argparse
import os, sys
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F
import gc

from tensorboardX import SummaryWriter
import _init_paths

from utils.metrics import AverageMeter, accuracy, multi_scores
from utils.checkpoint import Checkpoint

from networks.Inception_a import Inception_a as Inception_a
from dataset.loader import get_test_loader, get_train_loader

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')

"""The dataset file arguments.
"""
parser.add_argument('video', metavar='DIR', help='Directory to HMDB51 video')
parser.add_argument('frame', metavar='DIR', help='Directory to HMDB51 frame')
parser.add_argument('meta', metavar='DIR', help='Path to HMDB51 51 class meta information')
parser.add_argument('trainlist', metavar='DIR', help='Path to HMDB51 train list')
parser.add_argument('testlist', metavar='DIR', help='Path to HMDB51 test list')
parser.add_argument('--num-frame', default=10, type=int,
					help='Number of frames that extract from video')
parser.add_argument('--refresh', default=0, type=int,
					help='Refresh flag for clearing frames and create new one')


"""The training arguments including optimizer's arguments.
"""
# primary
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
					help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='N',
					help='optimizer\'s learning rate in training')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='N',
					help='optimizer\'s momentum in training')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float, metavar='N',
					help='optimizer\'s weight-decay in training')
parser.add_argument('-e', '--epoch', default=100, type=int, metavar='N',
					help='training epoch number')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (defult: 4)')
# secondary
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('-n', '--num-class', default=3, type=int, metavar='N',
					help='number of classes / categories')
parser.add_argument('--crop-size',default=299, type=int,
					help='crop size')
parser.add_argument('--scale-size',default=256, type=int,
					help='input size')


"""Record arguments.
"""
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
					help='print frequency (default: 10)')
parser.add_argument('--result-path', default='', type=str, metavar='PATH',
					help='path for saving result (default: none)')
parser.add_argument('--checkpoint-dir', default='', type=str, metavar='PATH',
					help='directory to load checkpoint (default: none)')
parser.add_argument('--checkpoint-name', default='', type=str, metavar='PATH',
					help='filename of checkpoint (default: none)')



def main():
	# global args, best_prec1
	args = parser.parse_args()
	print('\n====> Input Arguments')
	print(args)
	# Tensorboard writer.
	global writer
	writer = SummaryWriter(log_dir=args.result_path)


	# Create dataloader.
	print '\n====> Creating dataloader...'
	train_loader = get_train_loader(args)
	test_loader = get_test_loader(args)

	# Load Resnet_a network.
	print '====> Loading the network...'
	model = Inception_a(num_class=args.num_class, num_frame=args.num_frame, pretrained=True)

	"""Load checkpoint and weight of network.
	"""
	global cp_recorder
	if args.checkpoint_dir:
		cp_recorder = Checkpoint(args.checkpoint_dir, args.checkpoint_name)
		cp_recorder.load_checkpoint(model)
	
	model = nn.DataParallel(model)
	model.cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	cudnn.benchmark = True
	
	# optimizer = torch.optim.SGD(model.module.classifier.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
			
	# Train Resnet_a model.
	print '====> Training...'
	for epoch in range(cp_recorder.contextual['b_epoch'], args.epoch):
		_, _, prec_tri, rec_tri, ap_tri = train_eval(train_loader, test_loader, model, criterion, optimizer, args, epoch)
		top1_avg_val, loss_avg_val, prec_val, rec_val, ap_val = validate_eval(test_loader, model, criterion, args, epoch)

		# Print result.
		writer.add_scalars('mAP (per epoch)', {'train': np.nan_to_num(ap_tri).mean()}, epoch)
		writer.add_scalars('mAP (per epoch)', {'valid': np.nan_to_num(ap_val).mean()}, epoch)
		print('\n====> Scores')
		print('[Epoch {0}]:\n'
			'  Train:\n'
			'    Prec@1 {1}\n'
			'    Recall {2}\n'
			'    AP {3}\n'
			'    mAP {4:.3f}\n'
			'  Valid:\n'
			'    Prec@1 {5}\n'
			'    Recall {6}\n'
			'    AP {7}\n'
			'    mAP {8:.3f}\n'.format(epoch, 
				prec_tri, rec_tri, ap_tri, np.nan_to_num(ap_tri).mean(),
				prec_val, rec_val, ap_val, np.nan_to_num(ap_val).mean()))
		

		# Record.
		writer.add_scalars('Loss (per batch)', {'valid': loss_avg_val}, (epoch+1)*len(train_loader))
		writer.add_scalars('Prec@1 (per batch)', {'valid': top1_avg_val}, (epoch+1)*len(train_loader))
		writer.add_scalars('mAP (per batch)', {'valid': np.nan_to_num(ap_val).mean()}, (epoch+1)*len(train_loader))

		# Save checkpoint.
		cp_recorder.record_contextual({'b_epoch': epoch+1, 'b_batch': -1, 'prec': top1_avg_val, 'loss': loss_avg_val, 
			'class_prec': prec_val, 'class_recall': rec_val, 'class_ap': ap_val, 'mAP': np.nan_to_num(ap_val).mean()})
		cp_recorder.save_checkpoint(model)

def train_eval(train_loader, val_loader, model, criterion, optimizer, args, epoch, fnames=[]):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.train()
	
	# Fix resnet50.
	# model.module.resnet50.eval()

	end = time.time()
	scores = np.zeros((len(train_loader.dataset), args.num_class))
	labels = np.zeros((len(train_loader.dataset), ))

	# Checkpoint begin batch.
	b_batch=0
	if epoch == cp_recorder.contextual['b_epoch']:
		b_batch = cp_recorder.contextual['b_batch']+1
	
	for i, (frames, target) in enumerate(train_loader):
		# Jump to contextual batch
		if i < b_batch:
			continue
		target = target.cuda(async=True)
		frames = frames.cuda()
		output = model(frames)
		
		loss = criterion(output, target)
		losses.update(loss.item(), target.size(0))
		prec1 = accuracy(output.data, target)
		top1.update(prec1[0], target.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()
		
		if (i+1) % args.print_freq == 0:
			"""Every 10 batches, print on screen and print train information on tensorboard
			"""
			niter = epoch * len(train_loader) + i
			print('Train [Batch {0}/{1}|Epoch {2}/{3}]:  '
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
					'Loss {loss.val:.4f} ({loss.avg:.4f})  '
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
						i, len(train_loader), epoch, args.epoch, batch_time=batch_time,
						loss=losses, top1=top1))
			
		 	writer.add_scalars('Loss (per batch)', {'train-10b': loss.item()}, niter)
			writer.add_scalars('Prec@1 (per batch)', {'train-10b': prec1[0]}, niter)

		
		if (i+1) % (args.print_freq*10) == 0 :
			# Every 100 batches, print on screen and print validation information on tensorboard
			
			top1_avg_val, loss_avg_val, prec, recall, ap = validate_eval(val_loader, model, criterion, args, epoch)
			writer.add_scalars('Loss (per batch)', {'valid': loss_avg_val}, niter)
			writer.add_scalars('Prec@1 (per batch)', {'valid': top1_avg_val}, niter)
			writer.add_scalars('mAP (per batch)', {'valid': np.nan_to_num(ap).mean()}, niter)

			# Save checkpoint every 100 batches.
			cp_recorder.record_contextual({'b_epoch': epoch, 'b_batch': i, 'prec': top1_avg_val, 'loss': loss_avg_val, 
				'class_prec': prec, 'class_recall': recall, 'class_ap': ap, 'mAP': np.nan_to_num(ap).mean()})
			cp_recorder.save_checkpoint(model)
		
		# Record scores.
		output_f = F.softmax(output, dim=1)  # To [0, 1]
		output_np = output_f.data.cpu().numpy()
		labels_np = target.data.cpu().numpy()
		b_ind = i*args.batch_size
		e_ind = b_ind + min(args.batch_size, output_np.shape[0])
		scores[b_ind:e_ind, :] = output_np
		labels[b_ind:e_ind] = labels_np

	
	res_scores = multi_scores(scores, labels, ['precision', 'recall', 'average_precision'])
	print('Train [Epoch {0}/{1}]:  '
		'*Time {2:.2f}mins ({batch_time.avg:.2f}s)  '
		'*Loss {loss.avg:.4f}  '
		'*Prec@1 {top1.avg:.3f}'.format(epoch, args.epoch, batch_time.sum/60,
			batch_time=batch_time, loss=losses, top1=top1))
	
	return top1.avg, losses.avg, res_scores['precision'], res_scores['recall'], res_scores['average_precision']

def validate_eval(val_loader, model, criterion, args, epoch=None, fnames=[]):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.eval()

	end = time.time()
	scores = np.zeros((len(val_loader.dataset), args.num_class))
	labels = np.zeros((len(val_loader.dataset), ))
	for i, (frames, target) in enumerate(val_loader):
		with torch.no_grad():
			target = target.cuda(async=True)
			frames = frames.cuda()
			output = model(frames)
			
			loss = criterion(output, target)
			losses.update(loss.item(), target.size(0))
			prec1 = accuracy(output.data, target)
			top1.update(prec1[0], target.size(0))

			batch_time.update(time.time() - end)
			end = time.time()

			# Record scores.
			output_f = F.softmax(output, dim=1)  # To [0, 1]
			output_np = output_f.data.cpu().numpy()
			labels_np = target.data.cpu().numpy()
			b_ind = i*args.batch_size
			e_ind = b_ind + min(args.batch_size, output_np.shape[0])
			scores[b_ind:e_ind, :] = output_np
			labels[b_ind:e_ind] = labels_np
	
	print('Test [Epoch {0}/{1}]:  '
		'*Time {2:.2f}mins ({batch_time.avg:.2f}s)  '
		'*Loss {loss.avg:.4f}  '
		'*Prec@1 {top1.avg:.3f}'.format(epoch, args.epoch, batch_time.sum/60,
			batch_time=batch_time, top1=top1, loss=losses))

	res_scores = multi_scores(scores, labels, ['precision', 'recall', 'average_precision'])
	model.train()
	return top1.avg, losses.avg, res_scores['precision'], res_scores['recall'], res_scores['average_precision']


if __name__=='__main__':
	main()
