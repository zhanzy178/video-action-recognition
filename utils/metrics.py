# coding=utf-8
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

import sklearn.metrics as metrics
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def multi_scores(pre_scores, labels, options=['precision', 'recall', 'average_precision']):
	"""Make use of metrics.precision_score, metrics.recall_score, metrics.average_precision_score.
	
	The AP version in sklearn is VOC2010.
	"""
	pre_scores = np.nan_to_num(pre_scores)
	result = {}
	num_classes = pre_scores.shape[1]
	enc = OneHotEncoder()
	for op in options:
		if op == 'precision':
			scores = metrics.precision_score(labels, np.argmax(pre_scores, axis=1), labels=list(range(num_classes)), average=None)
		elif op == 'recall':
			scores = metrics.recall_score(labels, np.argmax(pre_scores, axis=1), labels=list(range(num_classes)), average=None)
		elif op == 'average_precision':
			# Labels one-hot encoding.
			enc.fit(np.arange(num_classes).reshape(-1, 1))
			labels_oh = enc.transform(labels.reshape(-1, 1)).toarray()
			scores = metrics.average_precision_score(labels_oh, pre_scores, average=None)
		else:
			continue
		result[op] = scores
	
	return result
