# coding-utf-8
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import HMDB51Dataset

def get_test_loader(args):
	"""Create dataset and return dataset loader of test Dataset.

	Args:
		TODO: Fill Args.

	Returns:
		test_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""
	
	video_dir = args.video
	frame_dir = args.frame
	meta_path = args.meta
	list_path = args.testlist
	num_frame = args.num_frame
	refresh = args.refresh != 0

	workers = args.workers
	batch_size = args.batch_size
	crop_size = args.crop_size

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	test_data_transform = transforms.Compose([
			transforms.Resize((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	test_set = HMDB51Dataset(video_dir, frame_dir, meta_path, list_path, input_transform = test_data_transform, num_frame=num_frame, refresh=refresh)
	test_loader = DataLoader(dataset=test_set, num_workers=workers,
							batch_size=batch_size, shuffle=False)
	return test_loader

def get_train_loader(args):
	"""Create dataset and return dataset loader of train Dataset.

	Args:
		TODO: Fill Args.

	Returns:
		train_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""
	video_dir = args.video
	frame_dir = args.frame
	meta_path = args.meta
	list_path = args.trainlist
	num_frame = args.num_frame
	refresh = args.refresh != 0

	workers = args.workers
	batch_size = args.batch_size
	crop_size = args.crop_size

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	train_data_transform = transforms.Compose([
			transforms.Resize((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	train_set = HMDB51Dataset(video_dir, frame_dir, meta_path, list_path, input_transform = train_data_transform, num_frame=num_frame, refresh=refresh)
	train_loader = DataLoader(dataset=train_set, num_workers=workers,
							batch_size=batch_size, shuffle=True)
	return train_loader

