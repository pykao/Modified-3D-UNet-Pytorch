import numpy as np
import SimpleITK as sitk
import cPickle as pickle
import os
import pprint
from dataset import load_dataset, BraTS2018List, BatchGenerator3D_random_sampling
import paths
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def testDataPreprocessing(pat_id = 0):

	train_dataset = load_BraTS2018_dataset()

	example_nda = train_dataset[pat_id]['data']

	print train_dataset[pat_id]['name'], train_dataset[pat_id]['type']

	t1_nda = example_nda[0, :]

	t1ce_nda = example_nda[1, :]

	t2_nda = example_nda[2, :]

	flair_nda = example_nda[3, :]

	seg_nda = example_nda[4, :]

	t1_img = sitk.GetImageFromArray(t1_nda)
	sitk.WriteImage(t1_img, './t1.nii.gz')
	t1ce_img = sitk.GetImageFromArray(t1ce_nda)
	sitk.WriteImage(t1ce_img, './t1ce.nii.gz')
	t2_img = sitk.GetImageFromArray(t2_nda)
	sitk.WriteImage(t2_img, './t2.nii.gz')
	flair_img = sitk.GetImageFromArray(flair_nda)
	sitk.WriteImage(flair_img, './flair.nii.gz')
	seg_img = sitk.GetImageFromArray(seg_nda)
	sitk.WriteImage(seg_img, './seg.nii.gz')

#data_path = paths.preprocessed_training_data_folder
#dataset = BraTS2018List(data_path=data_path, random_crop=(128, 128, 128))
#sample = dataset[42]
#print sample['data'].shape
#print sample['data'].type(), sample['seg'].type()
#print sample
#dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#for i_batch, sample_batch in enumerate(dataloader):
#    print(i_batch, sample_batch['name'], sample_batch['data'].size())

all_data = load_dataset()
keys_sorted = np.sort(all_data.keys())
#print all_data.keys()

train_idx, valid_idx = train_test_split(all_data.keys(), train_size = 0.9)

print train_idx

#train_keys = [keys_sorted[i] for i in train_idx]
#valid_keys = [keys_sorted[i] for i in valid_idx]

#print train_keys

#train_data = {i:all_data[i] for i in train_keys}
#valid_data = {i:all_data[i] for i in valid_keys}

#print len(train_data.keys()), len(valid_data.keys())

#data_gen_validation = BatchGenerator3D_random_sampling(valid_data, 2, num_batches=None, seed=False, patch_size=(128, 128, 128), convert_labels=True)
#for i_batch, sample_batch in enumerate(data_gen_validation):
#    print(i_batch, sample_batch['name'])
