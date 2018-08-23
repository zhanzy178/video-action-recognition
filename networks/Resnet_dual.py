# coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class Resnet_dual(nn.Module):
	def __init__(self, num_class = 51, num_frame = 10, num_object = 5, pretrained=False):
		super(Resnet_dual, self).__init__()
		# Args.
		self.num_class = num_class
		self.num_frame = num_frame
		self.num_object = num_object

		# First stream.
		self.resnet1 = models.resnet50(pretrained=pretrained)
		self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])

		self.cls1 = nn.Linear(2048, self.num_class)
		self.avg1 = nn.AvgPool2d((self.num_frame, 1), 1)
	
		# Second stream.
		self.resnet2 = models.resnet50(pretrained=pretrained)
		self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])
		
		self.cls2 = nn.Linear(2048*self.num_object, self.num_class)
		self.avg2 = nn.AvgPool2d((self.num_frame, 1), 1)
		
	# x1 = pu, x2 = p1, x3 = p2, x4 = bbox geometric info
	def forward(self, x, bboxes_list):
		# First stream.
		x = x.view(-1, 3, 224, 224)
		x = self.resnet1(x).view(-1, 2048)
		x = self.cls1(x).view(-1, 1, self.num_frame, self.num_class)
		x = self.avg2(x).view(-1, self.num_class)

		# Second stream.
		y = bboxes_list.view(-1, 3, 224, 224)
		y = self.resnet2(y).view(-1, self.num_object*2048)
		y = self.cls2(y).view(-1, 1, self.num_frame, self.num_class)
		y = self.avg2(y).view(-1, self.num_class)		

		return x+y
