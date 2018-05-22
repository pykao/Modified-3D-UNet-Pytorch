import os
import cPickle
import torch 
from torch.utils.data import Dataset
from multiprocessing import Pool
from shutil import copyfile

import SimpleITK as sitk
import numpy as np

import paths
from braintools import ReadImage
from utils import reshape_by_padding_upper_coords, random_crop_3D_image_batched
from data_loader import DataLoaderBase

class BraTS2018List(Dataset):
	def __init__(self, data_path, random_crop=None, to_tensor=True, convert_labels=True):
		"""
		Args:
			data_path (string): Directory with all the numpy files, pkl files and id_name_conversion.txt file
			transform (callable, optional): Optional transform to be applied on a sample
		"""
		self.data_path = data_path
		self.random_crop = random_crop
		assert len(self.random_crop) == 3, "The random crop size should be (x, y, z)"
		#self.transform = transform
		#self.sample_size = sample_size
		self.npy_names = sorted([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name)) and name.endswith('.npy')])
		self.pkl_names = sorted([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name)) and name.endswith('.pkl')])
		self.id_name_conversion = np.loadtxt(os.path.join(data_path, "id_name_conversion.txt"), dtype="str")
		self.to_tensor = to_tensor
		self.convert_labels = convert_labels
	def __getitem__(self, index):
		npy_data = np.load(os.path.join(self.data_path, self.npy_names[index]))
		#print os.path.join(self.data_path, self.npy_names[index])
		idxs = self.id_name_conversion[:, 1].astype(int)
		#print np.where(idxs == int(index))
		sample = {}
		with open(os.path.join(self.data_path, self.pkl_names[index]), 'r') as f:
			dp = cPickle.load(f)
		sample['name'] = self.id_name_conversion[np.where(idxs == int(index))[0][0], 0]
		sample['index'] = self.id_name_conversion[np.where(idxs == int(index))[0][0], 1]
		sample['type'] = self.id_name_conversion[np.where(idxs == int(index))[0][0], 2]
		sample['orig_shp'] = dp['orig_shp']
		#info['bbox_z'] = dp['bbox_z']
		#info['bbox_y'] = dp['bbox_y']
		#info['bbox_x'] = dp['bbox_x']
		sample['spacing'] = dp['spacing']
		sample['direction'] = dp['direction']
		sample['origin'] = dp['origin']
		image = npy_data[0:4, :]
		ori_label = npy_data[4, :]
		
		if self.convert_labels:
			new_label = convert_brats_seg(ori_label)
		else:
			new_label = np.copy(ori_label)


		if self.random_crop:
			"""
			Now only support random crop on certain size
			"""
			z, y, x = image.shape[1:]
			# shape of output
			new_z, new_y, new_x = (self.random_crop[0], self.random_crop[1], self.random_crop[2])
			if new_z == z:
				z += 1
			if new_y == y:
				y += 1
			if new_x == x:
				x += 1
			axial = np.random.randint(0, z - new_z)
			coronal = np.random.randint(0, y - new_y)
			sagittal = np.random.randint(0, x - new_x)
			image = image[:, axial: axial + new_z, coronal: coronal + new_y, sagittal: sagittal + new_x]
			new_label = new_label[axial: axial + new_z, coronal: coronal + new_y, sagittal: sagittal + new_x]
		
		if self.to_tensor:
			sample['data'] = torch.from_numpy(image)
			sample['seg'] = torch.from_numpy(new_label)
		else:
			sample['data'] = image
			sample['seg'] = new_label
		
		return sample

	def __len__(self):
		# of how many subjects it has
		return len(self.npy_names)


def extract_brain_region(image, brain_mask, background=0):
	''' find the boundary of the brain region, return the resized brain image and the index of the boundaries'''    
	brain = np.where(brain_mask != background)
	#print brain
	min_z = int(np.min(brain[0]))
	max_z = int(np.max(brain[0]))+1
	min_y = int(np.min(brain[1]))
	max_y = int(np.max(brain[1]))+1
	min_x = int(np.min(brain[2]))
	max_x = int(np.max(brain[2]))+1
	# resize image
	resizer = (slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
	return image[resizer], [[min_z, max_z], [min_y, max_y], [min_x, max_x]]


def cut_off_values_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
	if mask is None:
		mask = image != image[0, 0, 0]
	cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
	cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
	#print cut_off_lower, cut_off_upper
	res = np.copy(image)
	res[(res < cut_off_lower) & (mask !=0)] = cut_off_lower
	res[(res > cut_off_upper) & (mask !=0)] = cut_off_upper
	return res


def run(folder, out_folder, pat_id, name, return_if_no_seg=True, N4ITK = True):
	print pat_id, name
	if N4ITK: 
		t1_path = os.path.join(folder, "%s_t1_N4ITK_corrected.nii.gz" % name)
		t1ce_path = os.path.join(folder, "%s_t1ce_N4ITK_corrected.nii.gz" % name)
		t2_path = os.path.join(folder, "%s_t2_N4ITK_corrected.nii.gz" % name)
		flair_path = os.path.join(folder, "%s_flair_N4ITK_corrected.nii.gz" % name)
	if not N4ITK:
		t1_path = os.path.join(folder, "%s_t1.nii.gz" % name)
		t1ce_path = os.path.join(folder, "%s_t1ce.nii.gz" % name)
		t2_path = os.path.join(folder, "%s_t2.nii.gz" % name)
		flair_path = os.path.join(folder, "%s_flair.nii.gz" % name)
	seg_path = os.path.join(folder, "%s_seg.nii.gz" %name)
	if not os.path.isfile(t1_path):
		print "T1 file does not exist"
		return
	if not os.path.isfile(t1ce_path):
		print "T1ce file does not exist"
		return
	if not os.path.isfile(t2_path):
		print "T2 file does not exist"
		return
	if not os.path.isfile(flair_path):
		print "Flair file does not exist"
		return
	if not os.path.isfile(seg_path):
		if return_if_no_seg:
			print "Seg file does not exist"
			return
	t1_nda = ReadImage(t1_path)
	t1_img = sitk.ReadImage(t1_path)
	t1ce_nda = ReadImage(t1ce_path)
	t2_nda = ReadImage(t2_path)
	flair_nda = ReadImage(flair_path)
	#print t1_nda.shape, t1ce_nda.shape, t2_nda.shape, flair_nda.shape
	try:
		seg_nda = ReadImage(seg_path)
	except RuntimeError:
		seg_nda = np.zeros(t1_nda.shape, dtype = np.float32)
	except IOError:
		seg_nda = np.zeros(t1_nda.shape, dtype = np.float32)
	
	original_shape = t1_nda.shape

	brain_mask = (t1_nda != t1_nda[0, 0, 0]) & (t1ce_nda != t1ce_nda[0, 0, 0]) & (t2_nda != t2_nda[0, 0, 0]) & (flair_nda != flair_nda[0, 0, 0])

	resized_t1_nda, bbox = extract_brain_region(t1_nda, brain_mask, 0)
	resized_t1ce_nda, bbox1 = extract_brain_region(t1ce_nda, brain_mask, 0)
	resized_t2_nda, bbox2 = extract_brain_region(t2_nda, brain_mask, 0)
	resized_flair_nda, bbox3 = extract_brain_region(flair_nda, brain_mask, 0)
	resized_seg_nda, bbox4 = extract_brain_region(seg_nda, brain_mask, 0)
	assert bbox == bbox1 == bbox2 == bbox3 == bbox4
	assert resized_t1_nda.shape == resized_t1ce_nda.shape == resized_t2_nda.shape == resized_flair_nda.shape

	with open(os.path.join(out_folder, "%03.0d.pkl" % pat_id), 'w') as f:
		dp = {}
		dp['orig_shp'] = original_shape
		dp['bbox_z'] = bbox[0]
		dp['bbox_y'] = bbox[1]
		dp['bbox_x'] = bbox[2]
		dp['spacing'] = t1_img.GetSpacing()
		dp['direction'] = t1_img.GetDirection()
		dp['origin'] = t1_img.GetOrigin()
		cPickle.dump(dp, f)
	
	# setting the cut off threshold
	cut_off_threshold = 2.0

	t1_msk = resized_t1_nda != 0
	t1_tmp = cut_off_values_upper_lower_percentile(resized_t1_nda, t1_msk, cut_off_threshold, 100.0 - cut_off_threshold)
	normalized_resized_t1_nda = np.copy(resized_t1_nda)
	normalized_resized_t1_nda[t1_msk] = (resized_t1_nda[t1_msk] - t1_tmp[t1_msk].mean()) / t1_tmp[t1_msk].std()

	t1ce_msk = resized_t1ce_nda != 0
	t1ce_tmp = cut_off_values_upper_lower_percentile(resized_t1ce_nda, t1ce_msk, cut_off_threshold, 100.0 - cut_off_threshold)
	normalized_resized_t1ce_nda = np.copy(resized_t1ce_nda)
	normalized_resized_t1ce_nda[t1ce_msk] = (resized_t1ce_nda[t1ce_msk] - t1ce_tmp[t1ce_msk].mean()) / t1ce_tmp[t1ce_msk].std()

	t2_msk = resized_t2_nda != 0
	t2_tmp = cut_off_values_upper_lower_percentile(resized_t2_nda, t2_msk, cut_off_threshold, 100.0 - cut_off_threshold)
	normalized_resized_t2_nda = np.copy(resized_t2_nda)
	normalized_resized_t2_nda[t2_msk] = (resized_t2_nda[t2_msk] - t2_tmp[t2_msk].mean()) / t2_tmp[t2_msk].std()

	flair_msk = resized_flair_nda != 0
	flair_tmp = cut_off_values_upper_lower_percentile(resized_flair_nda, flair_msk, cut_off_threshold, 100.0 - cut_off_threshold)
	normalized_resized_flair_nda = np.copy(resized_flair_nda)
	normalized_resized_flair_nda[flair_msk] = (resized_flair_nda[flair_msk] - flair_tmp[flair_msk].mean()) / flair_tmp[flair_msk].std()

	shp = resized_t1_nda.shape
	#print shp

	new_shape = np.array([128, 128, 128])
	pad_size = np.max(np.vstack((new_shape, np.array(shp))), 0)
	#print pad_size
	new_t1_nda = reshape_by_padding_upper_coords(normalized_resized_t1_nda, pad_size, 0)
	new_t1ce_nda = reshape_by_padding_upper_coords(normalized_resized_t1ce_nda, pad_size, 0)
	new_t2_nda = reshape_by_padding_upper_coords(normalized_resized_t2_nda, pad_size, 0)
	new_flair_nda = reshape_by_padding_upper_coords(normalized_resized_flair_nda, pad_size, 0)
	new_seg_nda = reshape_by_padding_upper_coords(resized_seg_nda, pad_size, 0)
	#print new_t1_nda.shape, new_t1ce_nda.shape, new_t2_nda.shape, new_flair_nda.shape, new_seg_nda.shape
	number_of_data = 5
	#print [number_of_data]+list(new_t1_nda.shape)

	all_data = np.zeros([number_of_data]+list(new_t1_nda.shape), dtype=np.float32)
	#print all_data.shape
	all_data[0] = new_t1_nda
	all_data[1] = new_t1ce_nda
	all_data[2] = new_t2_nda
	all_data[3] = new_flair_nda
	all_data[4] = new_seg_nda
	np.save(os.path.join(out_folder, "%03.0d" % pat_id), all_data)


def run_star(args):
	return run(*args)


def run_preprocessing_BraTS2018_training(training_data_location=paths.raw_training_data_folder, folder_out=paths.preprocessed_training_data_folder, N4ITK=True):
	if not os.path.isdir(folder_out): os.mkdir(folder_out)
	ctr = 0
	id_name_conversion = []
	for f in ("HGG", "LGG"):
		fld = os.path.join(training_data_location, f)
		patients = os.listdir(fld)
		patients.sort()
		#print len(patients)
		fldrs = [os.path.join(fld, pt) for pt in patients]
		#print fldrs
		p = Pool(7)
		p.map(run_star, zip(fldrs, [folder_out]*len(patients), range(ctr, ctr + len(patients)), patients, len(patients) * [True], len(patients) * [N4ITK]))
		p.close()
		p.join()

		for i, j in zip(patients, range(ctr, ctr+len(patients))):
			id_name_conversion.append([i, j, f])
		ctr += (ctr+len(patients))
	id_name_conversion = np.vstack(id_name_conversion)
	np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
	copyfile(os.path.join(training_data_location, "survival_data.csv"), os.path.join(folder_out, "survival_data.csv"))

	 
def run_preprocessing_BraTS2018_validationOrTesting(original_data_location=paths.raw_validation_data_folder, folder_out=paths.preprocessed_validation_data_folder, N4ITK=True):
	if not os.path.isdir(folder_out): os.mkdir(folder_out)
	ctr = 0
	id_name_conversion = []
	patients = os.listdir(original_data_location)
	patients.sort()
	#print len(patients)
	fldrs = [os.path.join(fld, pt) for pt in patients]
	#print fldrs
	p = Pool(7)
	p.map(run_star, zip(fldrs, [folder_out]*len(patients), range(ctr, ctr + len(patients)), patients, len(patients) * [False], len(patients) * [N4ITK]))
	p.close()
	p.join()

	for i, j in zip(patients, range(ctr, ctr+len(patients))):
		id_name_conversion.append([i, j, 'unknown'])
	ctr += (ctr+len(patients))
	id_name_conversion = np.vstack(id_name_conversion)
	np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
	copyfile(os.path.join(training_data_location, "survival_data.csv"), os.path.join(folder_out, "survival_data.csv"))


def load_dataset(pat_ids = range(285), folder=paths.preprocessed_training_data_folder):
	id_name_conversion = np.loadtxt(os.path.join(folder, "id_name_conversion.txt"), dtype="str")
	#print id_name_conversion[0]
	idxs = id_name_conversion[:, 1].astype(int)
	#print idxs
	dataset = {}
	for pat in pat_ids:
		if os.path.isfile(os.path.join(folder, "%03.0d.npy" % pat)):
			dataset[pat] = {}
			dataset[pat]['data'] = np.load(os.path.join(folder, "%03.0d.npy" %pat), mmap_mode='r')
			dataset[pat]['idx'] = pat
			dataset[pat]['name'] = id_name_conversion[np.where(idxs == pat)[0][0], 0]
			dataset[pat]['type'] = id_name_conversion[np.where(idxs == pat)[0][0], 2]
			with open(os.path.join(folder, "%03.0d.pkl" % pat), 'r') as f:
				dp = cPickle.load(f)
			dataset[pat]['orig_shp'] = dp['orig_shp']
			dataset[pat]['bbox_z'] = dp['bbox_z']
			dataset[pat]['bbox_x'] = dp['bbox_x']
			dataset[pat]['bbox_y'] = dp['bbox_y']
			dataset[pat]['spacing'] = dp['spacing']
			dataset[pat]['direction'] = dp['direction']
			dataset[pat]['origin'] = dp['origin']
	return dataset


def convert_brats_seg(seg):
	new_seg = np.zeros(seg.shape, seg.dtype)
	new_seg[seg == 1] = 1
	new_seg[seg == 2] = 2
	# convert label 4 enhancing tumor to label 3
	new_seg[seg == 4] = 3
	return new_seg


def convert_to_brats_seg(seg):
	new_seg = np.zeros(seg.shape, seg.dtype)
	new_seg[seg == 1] = 2
	new_seg[seg == 2] = 4
	# convert label 3 back to label 4 enhancing tumor
	new_seg[seg == 3] = 4
	return new_seg

# Their code to generate 3D random batch for training
class BatchGenerator3D_random_sampling(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE, num_batches, seed, patch_size=(128, 128, 128), convert_labels=False):
        self.convert_labels = convert_labels
        self._patch_size = patch_size
        DataLoaderBase.__init__(self, data, BATCH_SIZE, num_batches, seed)

    def generate_train_batch(self):
        ids = np.random.choice(self._data.keys(), self.BATCH_SIZE)
        data = np.zeros((self.BATCH_SIZE, 4, self._patch_size[0], self._patch_size[1], self._patch_size[2]),
                        dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 1, self._patch_size[0], self._patch_size[1], self._patch_size[2]),
                       dtype=np.float32)
        types = []
        patient_names = []
        identifiers = []
        ages = []
        survivals = []
        for j, i in enumerate(ids):
            types.append(self._data[i]['type'])
            patient_names.append(self._data[i]['name'])
            identifiers.append(self._data[i]['idx'])
            # construct a batch, not very efficient
            data_all = self._data[i]['data'][None]
            if np.any(np.array(data_all.shape[2:]) - np.array(self._patch_size) < 0):
                new_shp = np.max(np.vstack((np.array(data_all.shape[2:])[None], np.array(self._patch_size)[None])), 0)
                data_all = resize_image_by_padding_batched(data_all, new_shp, 0)
            data_all = random_crop_3D_image_batched(data_all, self._patch_size)
            data[j, :] = data_all[0, :4]
            if self.convert_labels:
                seg[j, 0] = convert_brats_seg(data_all[0, 4])
            else:
                seg[j, 0] = data_all[0, 4]
            if 'survival' in self._data[i].keys():
                survivals.append(self._data[i]['survival'])
            else:
                survivals.append(np.nan)
            if 'age' in self._data[i].keys():
                ages.append(self._data[i]['age'])
            else:
                ages.append(np.nan)
        return {'data': data, 'seg': seg, "idx": ids, "grades": types, "identifiers": identifiers, "patient_names": patient_names, 'survival':survivals, 'age':ages}