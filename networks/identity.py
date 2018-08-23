import torch

class identity(torch.nn.Module):
	def __init__(self):
		super(identity, self).__init__()
	def forward(self, x):
		return x
