# coding=utf-8
import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random
import shutil

class HMDB51Dataset(data.Dataset):
	"""Human motion 51 Dataset Class.

	Args:
		TODO: Fill Args.

	Attributes:
		TODO: Fill Attributes.

	"""
	def __init__(self, video_dir, frame_dir, meta_path, list_path, input_transform = None, num_frame=10, refresh=False):
		super(HMDB51Dataset, self).__init__()
		
		self.video_dir = video_dir
		self.list_path = list_path
		self.num_frame = num_frame
		self.input_transform = input_transform

		# Get class label dict.
		self.class_dict = []
		with open(meta_path, 'r') as f:
			for line in f.readlines():
				self.class_dict.append(line.strip())

		# Get sample_list, label_list
		video_path_list = []
		self.sample_list = []
		self.label_list = []
		if self.sample_list == None:
			with open(self.list_path, 'r') as f:
				for line in f.readlines():
					video_name, label = line.split()
					video_path = os.path.join(self.video_dir, self.class_dict[int(label)], video_name)
					single_frame_dir = os.path.join(self.frame_dir, os.path.splitext(video_name)[0])

					self.label_list.append(label)
					video_path_list.append(video_path)
					self.sample_list.apend(single_frame_dir)
					
		
		# Check frames directory and create frames.
		if not os.path.isdir(frame_dir):
			os.mkdir(frame_dir)
		self.frame_dir  = os.path.join(frame_dir, str(num_frame))
		if not os.path.isdir(self.frame_dir):
			os.mkdir(self.frame_dir)
		self._create_frames(video_path_list, refresh=refresh)
		

			
	def _create_frames(self, video_path_list, refresh=False):
		if os.path.isdir(self.frame_dir):
			shutil.rmtree(self.frame_dir)
		os.mkdir(self.frame_dir)
		
		lines_len = len(video_path_list)
		for i, video_path in enumerate(video_path_list):
			single_frame_dir = self.sample_list[i]
			if not os.path.isdir(single_frame_dir):
				os.mkdir(single_frame_dir)

			if len(os.listdir(single_frame_dir)) != self.num_frame or refresh:
				# Extract frame.
				capturer = cv2.VideoCapture(video_path)

				frame_cnt = capturer.get(cv2.CAP_PROP_FRAME_COUNT)
				frame_splice = np.floor(np.linspace(0, frame_cnt, self.num_frame+1))
				frame_rand_ind = [np.random.randint(frame_splice[x], frame_splice[x+1]) for x in range(len(frame_splice)-1)]

				c = 0
				ind = 0
				while capturer.isOpened():   # Read every frame
					ret, vframe = capturer.read()
					if ret:
						if(frame_rand_ind[ind] == c):
							frame_item_path = os.path.join(single_frame_dir, str(ind)+'.jpg')
							cv2.imwrite(frame_item_path, vframe)
							ind += 1
						c = c + 1
					else:
						break
				capturer.release()
			print '/t====> %d/%d frames prepared'%(i, lines_len),
			
		print('...frames prepared')
				




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
			
			if frames == None:
				frame = torch.Tensor(size=(self.num_frame, img.size(0), img.size(1)))

			frame[i, :, :] = img
		
		return frames, self.label_list[index]

	def __len__(self):
		return len(self.names)
		# return 1024
