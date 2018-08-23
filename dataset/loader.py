# coding-utf-8
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from HMDB51Dataset import HMDB51Dataset_a, HMDB51Dataset_b, HMDB51Dataset_dual, HMDB51Dataset_sf
from KineticsDataset import KineticsDataset_a

def get_test_loader(args):
	"""Create dataset and return dataset loader of test Dataset.

	Args:
		TODO: Fill Args.

	Returns:
		test_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	test_data_transform = transforms.Compose([
			transforms.Resize((args.scale_size, args.scale_size)),
			transforms.CenterCrop((args.crop_size, args.crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip
	
	return get_loader(args, test_data_transform, 'test')



def get_train_loader(args):
	"""Create dataset and return dataset loader of train Dataset.

	Args:
		TODO: Fill Args.

	Returns:
		train_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_data_transform = transforms.Compose([
			transforms.Resize((args.scale_size, args.scale_size)),
			transforms.RandomCrop((args.crop_size, args.crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

		
	return get_loader(args, train_data_transform, 'train')


def get_loader(args, data_transform=None, mode='train'):
	"""Create dataset and return dataset loader of train Dataset.

	Args:
		TODO: Fill Args.

	Returns:
		train_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""
	video_dir = args.video
	frame_dir = args.frame
	meta_path = args.meta

	if mode == 'train':
		list_path = args.trainlist
	elif mode == 'test':
		list_path = args.testlist

	num_frame = args.num_frame
	refresh = args.refresh != 0

	workers = args.workers
	batch_size = args.batch_size


	dataset = args.dataset
	if dataset=='HMDB51Dataset_a':
		HMDB51Dataset=HMDB51Dataset_a
	elif dataset=='HMDB51Dataset_b':
		HMDB51Dataset=HMDB51Dataset_b
	elif dataset=='HMDB51Dataset_dual':
		HMDB51Dataset=HMDB51Dataset_dual
	elif dataset=='KineticsDataset_a':
                KineticsDataset = KineticsDataset_a
	elif dataset=='HMDB51Dataset_sf':
		HMDB51Dataset=HMDB51Dataset_sf
	
	if dataset=='HMDB51Dataset_a' or dataset=='HMDBD51Dataset_b' or dataset=='HMDB51Dataset_sf':
		v_dataset = HMDB51Dataset(video_dir, frame_dir, meta_path, list_path, input_transform = data_transform, num_frame=num_frame, refresh=refresh)
	elif dataset=='HMDB51Dataset_dual':
		object_transform =  transforms.Compose([
				transforms.Resize((crop_size, crop_size)),
				transforms.ToTensor(),
				normalize])
		v_dataset = HMDB51Dataset(video_dir, frame_dir, meta_path, list_path, args.train_objectlist, input_transform = data_transform, object_transform=object_transform, num_frame=num_frame, num_object=args.num_object, refresh=refresh)
	elif dataset=='KineticsDataset_a':
                v_dataset = KineticsDataset(video_dir+'_'+mode, frame_dir, meta_path, list_path, input_transform = data_transform, num_frame=num_frame, refresh=refresh)

	loader = DataLoader(dataset=v_dataset, num_workers=workers,
							batch_size=batch_size, shuffle=True)
	return loader

