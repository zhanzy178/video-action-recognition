# coding=utf-8
import os, sys, time
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random
import shutil
import pandas as pd

class KineticsDataset_a(data.Dataset):
	"""Kinetics Action Recognition Dataset Class.

	Args:
		TODO: Fill Args.

	Attributes:
		TODO: Fill Attributes.

	"""
	def __init__(self, video_dir, frame_dir, meta_path, list_path, input_transform = None, num_frame=10, refresh=False):
		super(KineticsDataset_a, self).__init__()
		
		self.video_dir = video_dir
		self.list_path = list_path
		self.num_frame = num_frame
		self.input_transform = input_transform

		# Get class label dict.
		self.class_dict = {}
		with open(meta_path, 'r') as f:
			for i, line in enumerate(f.readlines()):
				self.class_dict[line.strip()] = i
		
		# Check frames directory and create frames.
		if not os.path.isdir(frame_dir):
			os.mkdir(frame_dir)
		self.frame_dir  = os.path.join(frame_dir, str(num_frame))
		if not os.path.isdir(self.frame_dir):
			os.mkdir(self.frame_dir)
		
		# Get sample_list, label_list
		video_path_list = []
		self.sample_list = []
		self.label_list = []
		anno_list = pd.read_csv(list_path)
		for a in range(len(anno_list)):
			video_name = anno_list.iloc[a, 1]+'.mp4'
			label = anno_list.iloc[a, 0]
			video_path = os.path.join(self.video_dir, video_name)
			if os.path.isfile(video_path):
				single_frame_dir = os.path.join(self.frame_dir, os.path.splitext(video_name)[0])

				self.label_list.append(self.class_dict[label])
				video_path_list.append(video_path)
				self.sample_list.append(single_frame_dir)
		
		# Extract frames.
		start = time.time()
		print('====> Checking frames in "%s"'%list_path)
		self._extract_frames(video_path_list, refresh=refresh)
		print('...costs %fs'%(time.time()-start))
		

			
	def _extract_frames(self, video_path_list, refresh=False):
		lines_len = len(video_path_list)
		error_video = []
		for i, video_path in enumerate(video_path_list):
			single_frame_dir = self.sample_list[i]
			if not os.path.isdir(single_frame_dir):
				os.mkdir(single_frame_dir)

			if len(os.listdir(single_frame_dir)) != self.num_frame or refresh:
				# Extract frame.
				capturer = cv2.VideoCapture(video_path)

				frame_cnt = capturer.get(cv2.CAP_PROP_FRAME_COUNT)
				frame_splice = np.floor(np.linspace(0, frame_cnt-1, self.num_frame+1))
				#if frame_cnt-1 < self.num_frame:
				#	frame_rand_ind = [frame_splice[x] for x in range(len(frame_splice)-1)]
				#else:
				#	frame_rand_ind = [np.random.randint(frame_splice[x], frame_splice[x+1]) for x in range(len(frame_splice)-1)]
				frame_rand_ind = [frame_splice[x] for x in range(len(frame_splice)-1)]

				c = 0
				ind = 0
				while capturer.isOpened():   # Read every frame
					ret, vframe = capturer.read()
					if ret:
						while frame_rand_ind[ind] == c:
							frame_item_path = os.path.join(single_frame_dir, str(ind)+'.jpg')
							cv2.imwrite(frame_item_path, vframe)
							ind += 1
							if ind == len(frame_rand_ind):
								break
						if ind == len(frame_rand_ind):
							break
						c = c + 1
					else:
						break
				capturer.release()
				# Error: moov atom not found python cv2
				if (c == 0):
					error_video.append(i)
			print '\r====> %d/%d frames prepared'%(i, lines_len),
		
		for err_ind in error_video:
			self.sample_list.pop(err_ind)
			video_path_list.pop(err_ind)
			self.label_list.pop(err_ind)
		print('\n...%d frames prepared'%i)
				




	def __getitem__(self, index):
		"""Get dataset item by index.

		Args:
			index: The index of one sample in dataset.
		
		Returns:
			TODO: Fill Returns.

		"""
		frames = None
		for i in range(self.num_frame):
			img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB') # convert gray to rgb
			if self.input_transform:
				img = self.input_transform(img)
			if type(frames) == type(None):
				frames = torch.Tensor(size=(self.num_frame, img.size(0), img.size(1), img.size(2)))
			
			frames[i, :, :, :] = img
		
		return frames, self.label_list[index]

	def __len__(self):
		return len(self.label_list)
		# return 1024


