# coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class Inception_a(nn.Module):
	def __init__(self, num_class = 51, num_frame = 10, pretrained=False):
		super(Inception_a, self).__init__()
		# Args.
		self.num_class = num_class
		self.num_frame = num_frame

		# Layers.
		self.inception = models.inception_v3(pretrained=pretrained)
		self.inception = nn.Sequential(*list(self.inception.children())[:-1])

		self.classifier = nn.Linear(2048, self.num_class)
		self.avgpool = nn.AvgPool2d((self.num_frame, 1), 1)
		

	# x1 = pu, x2 = p1, x3 = p2, x4 = bbox geometric info
	def forward(self, x):
		x = x.view(-1, 3, 299, 299)
		x = self.inception(x).view(-1, 2048)
		x = self.classifier(x).view(-1, 1, self.num_frame, self.num_class)
		x = self.avgpool(x).view(-1, self.num_class)
		return x
