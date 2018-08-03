# coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class LSTM_r(nn.Module):
	def __init__(self, num_class = 51, num_frame = 10, pretrained=False):
		super(LSTM_r, self).__init__()
		# Args.
		self.num_class = num_class
		self.num_frame = num_frame

		# Layers.
		self.resnet50 = models.resnet50(pretrained=pretrained)
		self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
		
		self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True)
		self.classifier = nn.Linear(2048, self.num_class)	

	# x1 = pu, x2 = p1, x3 = p2, x4 = bbox geometric info
	def forward(self, x):
		x = x.view(-1, 3, 224, 224)
		x = self.resnet50(x).view(-1, self.num_frame, 2048)
		_, (h_n, _) = self.lstm(x)
		x = self.classifier(h_n.view(-1, 2048))
		return x
