# coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from identity import identity

class Inception_a(nn.Module):
	def __init__(self, num_class = 51, num_frame = 10, pretrained=False):
		super(Inception_a, self).__init__()
		# Args.
		self.num_class = num_class
		self.num_frame = num_frame

		# Layers.
		self.inception = models.inception_v3(pretrained=pretrained)
		
		# Conv Pooling
		self.inception.fc = identity()	
		self.avgpool = nn.AvgPool2d((self.num_frame, 1), 1)
		self.fc = nn.Linear(2048, self.num_class, bias=True)

		# Late Pooling
		# self.inception.fc = nn.Linear(2048, self.num_class, bias=True)
		# self.avgpool = nn.AvgPool2d((self.num_frame, 1), 1)
		

	def forward(self, x):
		x = x.view(-1, 3, 299, 299)
		
		if self.training: 
			x, _ = self.inception(x)
		else:
			x = self.inception(x)

		# Conv Pooling
		x = x.view(-1, 1, self.num_frame, 2048)
		x = self.avgpool(x).view(-1, 2048)
		x = self.fc(x)
		
		# Late Pooling
		# x = x.view(-1, 1, self.num_frame, self.num_class)
		# x = self.avgpool(x).view(-1, self.num_class)
		return x
