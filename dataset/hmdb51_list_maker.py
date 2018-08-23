# coding=utf-8
import numpy as np
import os
import argparse

"""
$ python hmdb51_list_maker.py data/HMDB51
"""

parser = argparse.ArgumentParser(description='HMDB51 test/train list maker')
parser.add_argument('data', metavar='DIR', help='path to dataset')

def make_test_train_list():
	dataset_path = parser.parse_args().data
	split_path = os.path.join(dataset_path, 'split', 'testTrainMulti_7030_splits')
	# Make meta.
	split_file = os.listdir(split_path)
	class_meta = []
	for file_ in split_file:
		if file_.find('_test_split1.txt') != -1:
			class_meta.append(file_.replace('_test_split1.txt', ''))

	with open(os.path.join(dataset_path, 'meta.txt'), 'w') as f:
		for class_ in class_meta:
			f.write(class_+'\n')

	# Make list.	
	train_list = []
	train_label = []
	test_list = []
	test_label = []
	for i, c in enumerate(class_meta):
		for split in range(1, 2):
			filepath = os.path.join(split_path, c + '_test_split' + str(split) + '.txt')
			with open(filepath, 'r') as f:
				for line in f.readlines():
					video_name, id = line.split()
					if id == '1':
						# if video_name in train_list: continue
						train_list.append(video_name)
						train_label.append(i)
					elif id == '2':
						# if video_name in test_list: continue
						test_list.append(video_name)
						test_label.append(i)
	

	
	r_train_ind = np.arange(len(train_list))
	np.random.shuffle(r_train_ind)
	r_test_ind = np.arange(len(test_list))
	np.random.shuffle(r_test_ind)
	
	with open(os.path.join(dataset_path, 'test_list.txt'), 'w') as f:
		for i in range(len(test_label)):
			f.write(test_list[r_test_ind[i]]+' '+str(test_label[r_test_ind[i]])+'\n')

	with open(os.path.join(dataset_path, 'train_list.txt'), 'w') as f:
		for i in range(len(train_label)):
			f.write(train_list[r_train_ind[i]]+' '+str(train_label[r_train_ind[i]])+'\n')

	print('Class num is %d'%len(class_meta))
	print('Test list length is %d'%len(test_list))
	print('Train list length is %d'%len(train_list))


if __name__ == '__main__':
	make_test_train_list()
