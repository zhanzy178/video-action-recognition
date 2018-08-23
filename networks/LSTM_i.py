# coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from identity import identity

class LSTM_i(nn.Module):
	def __init__(self, num_class = 51, num_frame = 10, pretrained=False):
		super(LSTM_i, self).__init__()
		# Args.
		self.num_class = num_class
		self.num_frame = num_frame

		# Layers.
		self.inception = models.inception_v3(pretrained=pretrained)
		self.inception.fc = identity()

		self.lstm = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True)
		self.classifier = nn.Linear(512, self.num_class)
		self.avgpool = nn.AvgPool2d((self.num_frame, 1), 1)

	# x1 = pu, x2 = p1, x3 = p2, x4 = bbox geometric info
	def forward(self, x):
		x = x.view(-1, 3, 299, 299)
		if self.training:
			x, _ = self.inception(x)
			x = x.view(-1, self.num_frame, 2048)
		else:
			x = self.inception(x).view(-1, self.num_frame, 2048)
		output, (h_n, _) = self.lstm(x)
		output = output.contiguous()
		
		if self.training:
			x = self.classifier(output.view(-1, 512)).view(-1, 1, self.num_frame, self.num_class)
			return self.avgpool(x).view(-1, self.num_class)
		else:
			x = self.classifier(output[:, -1, :].view(-1, 512)).view(-1, self.num_class)
			return x
