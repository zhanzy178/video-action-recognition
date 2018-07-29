# coding=utf-8
import torch
import os
class Checkpoint:
	def __init__(self, checkpoint_dir='', filename=''):
		self.contextual = {}
		self.contextual['b_epoch'] = 0
		self.contextual['b_batch'] = -1
		self.contextual['prec'] = 0
		self.contextual['loss'] = 0
		self.contextual['class_prec'] = []
		self.contextual['class_recall'] = []
		self.contextual['class_ap'] = []
		self.contextual['mAP'] = 0
		self.checkpoint_dir = checkpoint_dir
		self.filename=filename
		self.best_prec1 = 0
		self.best_loss = -1
		self.best_p = False
		self.best_l = False
	
	def record_contextual(self, contextual):
		self.contextual = contextual
		if self.contextual['prec'] > self.best_prec1:
			self.best_p = True
			self.best_prec1 = self.contextual['prec']
		else:
			self.best_p = False

		if self.contextual['loss'] < self.best_loss or self.best_loss == -1:
			self.best_l = True
			self.best_loss = self.contextual['loss']
		else:
			self.best_l = False


	def save_checkpoint(self, model):
		path = os.path.join(self.checkpoint_dir, self.filename)

		# Save contextual.
		torch.save(self.contextual, path+'_contextual.pth')
		print('...Contextual saved')

		# Save model.
		torch.save(model.state_dict(), path+'.pth')
		print('...Model saved')

		if (self.best_p):
			torch.save(self.contextual, path+'_contextual_best_p.pth')
			torch.save(model.state_dict(), path+'_best_p.pth')
			print('...Max precision model and contextual saved')

		if (self.best_l):
			torch.save(self.contextual, path+'_contextual_best_l.pth')
			torch.save(model.state_dict(), path+'_best_l.pth')
			print('...Min loss model and contextual saved')


	def load_checkpoint(self, model):
		path = os.path.join(self.checkpoint_dir, self.filename)

		# Load contextual.
		if path and os.path.isfile(path+'_contextual.pth'):
			print("====> Loading checkpoint contextual '{}'...".format(path+'_contextual.pth'))
			self.contextual = torch.load(path+'_contextual.pth')

			# Update best prec.
			if self.contextual['prec'] > self.best_prec1:
				self.best_p = True
				self.best_prec1 = self.contextual['prec']
			else:
				self.best_p = False

			# Update best loss.
			if self.contextual['loss'] < self.best_loss or self.best_loss == -1:
				self.best_l = True
				self.best_loss = self.contextual['loss']
			else:
				self.best_l = False
		else:
			print("====> No checkpoint contextual at '{}'".format(path+'_contextual.pth'))

		# Load model.
		if path and os.path.isfile(path+'.pth'):
			print("====> Loading model '{}'...".format(path+'.pth'))
			state_dict = torch.load(path+'.pth')
			keys = model.state_dict().keys()
			new_state_dict = {}
			for i, k in enumerate(state_dict.keys()):
				new_state_dict[keys[i]] = state_dict[k]
			model.load_state_dict(new_state_dict)
		else:
			print("====> No pretrain model at '{}'".format(path+'.pth'))
