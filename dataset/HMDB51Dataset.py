# coding=utf-8
import os, sys, time, math
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random
import shutil

class HMDB51Dataset_a(data.Dataset):
	"""Human motion 51 Dataset Class.

	Args:
		TODO: Fill Args.

	Attributes:
		TODO: Fill Attributes.

	"""
	def __init__(self, video_dir, frame_dir, meta_path, list_path, input_transform = None, num_frame=10, refresh=False):
		super(HMDB51Dataset_a, self).__init__()
		
		self.video_dir = video_dir
		self.list_path = list_path
		self.num_frame = num_frame
		self.input_transform = input_transform

		# Get class label dict.
		self.class_dict = []
		with open(meta_path, 'r') as f:
			for line in f.readlines():
				self.class_dict.append(line.strip())
		
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
		with open(self.list_path, 'r') as f:
			for line in f.readlines():
				video_name, label = line.split()
				video_path = os.path.join(self.video_dir, self.class_dict[int(label)], video_name)
				single_frame_dir = os.path.join(self.frame_dir, os.path.splitext(video_name)[0])

				self.label_list.append(int(label))
				video_path_list.append(video_path)
				self.sample_list.append(single_frame_dir)
		
		# Extract frames.
		start = time.time()
		print('====> Checking frames in "%s"'%list_path)
		self._extract_frames(video_path_list, refresh=refresh)
		print('...costs %fs'%(time.time()-start))
		

			
	def _extract_frames(self, video_path_list, refresh=False):
		lines_len = len(video_path_list)
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
			print '\r====> %d/%d frames prepared'%(i, lines_len),
			
		print('\n...%d frames prepared'%i)
				




	def __getitem__(self, index):
		"""Get dataset item by index.

		Args:
			index: The index of one sample in dataset.
		
		Returns:
			TODO: Fill Returns.

		"""
		# hflip = random.random() < 0.5
		frames = None
		for i in range(self.num_frame):
			img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB') # convert gray to rgb
			# if not hflip:
			# 	img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB') # convert gray to rgb
			# else:
			# 	img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT)
			
			if self.input_transform:
				img = self.input_transform(img)
			if type(frames) == type(None):
				frames = torch.Tensor(size=(self.num_frame, img.size(0), img.size(1), img.size(2)))
			
			frames[i, :, :, :] = img
		
		return frames, self.label_list[index]

	def __len__(self):
		return len(self.label_list)
		# return 1024

class HMDB51Dataset_b(data.Dataset):
	"""Human motion 51 Dataset Class.

	Args:
		TODO: Fill Args.

	Attributes:
		TODO: Fill Attributes.

	"""
	def __init__(self, video_dir, frame_dir, meta_path, list_path, input_transform = None, num_frame=10, refresh=False):
		super(HMDB51Dataset_b, self).__init__()
		
		self.video_dir = video_dir
		self.list_path = list_path
		self.num_frame = num_frame
		self.input_transform = input_transform

		# Get class label dict.
		self.class_dict = []
		with open(meta_path, 'r') as f:
			for line in f.readlines():
				self.class_dict.append(line.strip())
		
		# Check frames directory and create frames.
		if not os.path.isdir(frame_dir):
			os.mkdir(frame_dir)
		self.frame_dir  = os.path.join(frame_dir, 'full')
		if not os.path.isdir(self.frame_dir):
			os.mkdir(self.frame_dir)
		
		# Get sample_list, label_list
		video_path_list = []
		self.sample_list = []
		self.label_list = []
		with open(self.list_path, 'r') as f:
			for line in f.readlines():
				video_name, label = line.split()
				video_path = os.path.join(self.video_dir, self.class_dict[int(label)], video_name)
				single_frame_dir = os.path.join(self.frame_dir, os.path.splitext(video_name)[0])

				self.label_list.append(int(label))
				video_path_list.append(video_path)
				self.sample_list.append(single_frame_dir)
		
		# Extract frames.
		start = time.time()
		print('====> Checking frames in "%s"'%list_path)
		self._extract_frames(video_path_list, refresh=refresh)
		print('...costs %fs'%(time.time()-start))
		

			
	def _extract_frames(self, video_path_list, refresh=False):
		lines_len = len(video_path_list)
		self.clip_ind = []
		for i, video_path in enumerate(video_path_list):
			single_frame_dir = self.sample_list[i]
			if not os.path.isdir(single_frame_dir):
				os.mkdir(single_frame_dir)

			if len(os.listdir(single_frame_dir)) == 0 or refresh:
				# Extract frame.
				capturer = cv2.VideoCapture(video_path)
				frame_cnt = capturer.get(cv2.CAP_PROP_FRAME_COUNT)

				c = 0
				while capturer.isOpened():   # Read every frame
					ret, vframe = capturer.read()
					if ret:
						frame_item_path = os.path.join(single_frame_dir, str(c)+'.jpg')
						cv2.imwrite(frame_item_path, vframe)
						ind += 1
						c = c + 1
					else:
						break
				capturer.release()
			for p in range(0, len(os.listdir(single_frame_dir)), self.num_frame):
				self.clip_ind.append([i, p])
			print '\r====> %d/%d frames prepared'%(i, lines_len),
		print('\n...%d frames prepared'%i)
		self.clip_ind = np.array(self.clip_ind)
				

	def __getitem__(self, index):
		"""Get dataset item by index.

		Args:
			index: The index of one sample in dataset.
		
		Returns:
			TODO: Fill Returns.

		"""
		frames = None
		video_ind = self.clip_ind[index][0]	
		for i in range(self.num_frame):
			frame_ind = (self.clip_ind[index][1]+i) % len(os.listdir(self.sample_list[video_ind]))
			img = Image.open(os.path.join(self.sample_list[video_ind], str(frame_ind)+'.jpg')).convert('RGB') # convert gray to rgb
			if self.input_transform:
				img = self.input_transform(img)
			if type(frames) == type(None):
				frames = torch.Tensor(size=(self.num_frame, img.size(0), img.size(1), img.size(2)))
			
			frames[i, :, :, :] = img
		
		return frames, self.label_list[video_ind]

	def __len__(self):
		return len(self.clip_ind)
		# return 1024

class HMDB51Dataset_dual(data.Dataset):
	"""Human motion 51 Dataset Class.

	Args:
		TODO: Fill Args.

	Attributes:
		TODO: Fill Attributes.

	"""
	def __init__(self, video_dir, frame_dir, meta_path, list_path, object_list_path, input_transform = None, object_transform = None, num_frame=10, num_object=5, refresh=False):
		super(HMDB51Dataset_dual, self).__init__()
		
		self.video_dir = video_dir
		self.list_path = list_path
		self.object_list_path = object_list_path
		self.num_frame = num_frame
		self.num_object = num_object
		self.input_transform = input_transform
		self.object_transform = object_transform

		# Get class label dict.
		self.class_dict = []
		with open(meta_path, 'r') as f:
			for line in f.readlines():
				self.class_dict.append(line.strip())
		

		self.frame_dir  = os.path.join(frame_dir, str(self.num_frame))
		# Get sample_list, label_list
		self.sample_list = []
		self.label_list = []
		with open(self.list_path, 'r') as f:
			for line in f.readlines():
				video_name, label = line.split()
				video_path = os.path.join(self.video_dir, self.class_dict[int(label)], video_name)
				single_frame_dir = os.path.join(self.frame_dir, os.path.splitext(video_name)[0])

				self.label_list.append(int(label))
				self.sample_list.append(single_frame_dir)
					
		self.bbox_list = torch.load(self.object_list_path)	

	def __getitem__(self, index):
		"""Get dataset item by index.

		Args:
			index: The index of one sample in dataset.
		
		Returns:
			TODO: Fill Returns.

		"""
		frames = None
		objects = None
		for i in range(self.num_frame):
			img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB') # convert gray to rgb
			for o in range(self.num_object):
				bbox = self.bbox_list[index*i, o, :]
				object_ = img.crop(bbox)
				if self.object_transform:
					object_ = self.object_transform(object_)
				if type(objects) == type(None):
					objects = torch.Tensor(size=(self.num_frame*self.num_object, object_.size(0), object_.size(1), object_.size(2)))
				objects[i*self.num_object+o, :, :, :] = object_
			
			if self.input_transform:
				img = self.input_transform(img)
			
			if type(frames) == type(None):
				frames = torch.Tensor(size=(self.num_frame, img.size(0), img.size(1), img.size(2)))
			
			frames[i, :, :, :] = img
		
		return frames, objects,  self.label_list[index]


	def __len__(self):
		return len(self.sample_list)
		# return 1024


class HMDB51Dataset_sf(data.Dataset):
	"""Human motion 51 Dataset Class.

	Args:
		TODO: Fill Args.

	Attributes:
		TODO: Fill Attributes.

	"""
	def __init__(self, video_dir, frame_dir, meta_path, list_path, input_transform = None, num_frame=10, refresh=False):
		super(HMDB51Dataset_sf, self).__init__()
		
		self.video_dir = video_dir
		self.list_path = list_path
		# self.num_frame = num_frame
		self.num_frame = 16
		self.input_transform = input_transform

		# Get class label dict.
		self.class_dict = []
		with open(meta_path, 'r') as f:
			for line in f.readlines():
				self.class_dict.append(line.strip())
		
		# Check frames directory and create frames.
		if not os.path.isdir(frame_dir):
			os.mkdir(frame_dir)
		self.frame_dir  = os.path.join(frame_dir, str(self.num_frame))
		if not os.path.isdir(self.frame_dir):
			os.mkdir(self.frame_dir)
		
		# Get sample_list, label_list
		video_path_list = []
		self.sample_list = []
		self.label_list = []
		with open(self.list_path, 'r') as f:
			for line in f.readlines():
				video_name, label = line.split()
				video_path = os.path.join(self.video_dir, self.class_dict[int(label)], video_name)
				single_frame_dir = os.path.join(self.frame_dir, os.path.splitext(video_name)[0])
				
				self.label_list.append(int(label))
				video_path_list.append(video_path)
				self.sample_list.append(single_frame_dir)
		
		# Extract frames.
		start = time.time()
		print('====> Checking frames in "%s"'%list_path)
		self._extract_frames(video_path_list, refresh=refresh)
		print('...costs %fs'%(time.time()-start))
		

			
	def _extract_frames(self, video_path_list, refresh=False):
		lines_len = len(video_path_list)
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
			print '\r====> %d/%d frames prepared'%(i, lines_len),
			
		print('\n...%d frames prepared'%i)
				




	def __getitem__(self, index):
		"""Get dataset item by index.

		Args:
			index: The index of one sample in dataset.
		
		Returns:
			TODO: Fill Returns.

		"""
		# hflip = random.random() < 0.5
		frame_ind = index % self.num_frame
		video_ind = int(math.floor(index / self.num_frame))
		img = Image.open(os.path.join(self.sample_list[video_ind], str(frame_ind)+'.jpg')).convert('RGB') # convert gray to rgb
		# if not hflip:
		# 	img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB') # convert gray to rgb
		# else:
		# 	img = Image.open(os.path.join(self.sample_list[index], str(i)+'.jpg')).convert('RGB').transpose(Image.FLIP_LEFT_RIGHT)
		if self.input_transform:
			img = self.input_transform(img)
		frames = torch.Tensor(size=(1, img.size(0), img.size(1), img.size(2)))	
		frames[0, :, :, :] = img
		
		return frames, self.label_list[video_ind]

	def __len__(self):
		return len(self.label_list)*self.num_frame
		# return 1024


