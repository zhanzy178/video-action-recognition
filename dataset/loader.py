import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SRDataset

def get_test_loader(args):
	"""Create dataset and return dataset loader of test Dataset.

	Args:
		args.data_dir: The directory of the dataset image.
		args.objects_dir: The directory of the ROI object extracted from image by Faster RCNN.
		args.test_list: The file path of annotation list with the unit of content: image_id, box1, box2, label.
		args.scale_size: Scale size of transform.
		args.crop_size: Crop size of trnasform.
		args.workers: The workers number.
		args.batch_size: The batch size to load sample.

	Returns:
		test_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""

	data_dir = args.data
	test_list = args.testlist
	objects_dir = args.objects
	scale_size = args.scale_size
	crop_size = args.crop_size
	workers = args.workers
	batch_size = args.batch_size

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	test_data_transform = transforms.Compose([
			transforms.Resize((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	test_full_transform = transforms.Compose([
			transforms.Resize((448, 448)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	test_set = SRDataset(data_dir, objects_dir, test_list, test_data_transform, test_full_transform)
	test_loader = DataLoader(dataset=test_set, num_workers=workers,
							batch_size=batch_size, shuffle=False)
	return test_loader

def get_train_loader(args):
	"""Create dataset and return dataset loader of train Dataset.

	Args:
		args.data_dir: The directory of the dataset image.
		args.objects_dir: The directory of the ROI object extracted from image by Faster RCNN.
		args.train_list: The file path of annotation list with the unit of content: image_id, box1, box2, label.
		args.scale_size: Scale size of transform.
		args.crop_size: Crop size of trnasform.
		args.workers: The workers number.
		args.batch_size: The batch size to load sample.

	Returns:
		train_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""
	data_dir = args.data
	train_list = args.trainlist
	objects_dir = args.objects
	scale_size = args.scale_size
	crop_size = args.crop_size
	workers = args.workers
	batch_size = args.batch_size

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	train_data_transform = transforms.Compose([
			transforms.Resize((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	train_full_transform = transforms.Compose([
			transforms.Resize((448, 448)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	train_set = SRDataset(data_dir, objects_dir, train_list, train_data_transform, train_full_transform )
	train_loader = DataLoader(dataset=train_set, num_workers=workers,
							batch_size=batch_size, shuffle=True)
	return train_loader

