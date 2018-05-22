#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:06:53 2018

@author: pkao

This code contains several tools for BRATS2017 database
"""

import os
import numpy as np
import SimpleITK as sitk

#def Brats2017FASTSegPathes(bratsPath):
#	''' This code returns the FAST segment maps'''
#	seg_filepathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
#					for name in files if 'seg' in name and 'FAST' in name and 'MNI152' not in name and name.endswith('.nii.gz')]
#	seg_filepathes.sort()

#	return seg_filepathes



#def BrainParcellationPathes(bratsPath, brainParcellationName):
#	'''This function gives you the location of brain parcellation mask in patient space '''
#	brain_parcellation_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
#	for name in files if brainParcellationName in name and 'lab' not in name and 'threshold' in root and name.endswith('.nii.gz')]
#	brain_parcellation_pathes.sort()
#	return brain_parcellation_pathes

def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    

def ModalityMaximum(filepathes):
    ''' This code returns the maximum value for MR images'''
    modality_maximum = 0
    for i in range(len(filepathes)):
        temp_img = sitk.ReadImage(filepathes[i])
        temp_nda = sitk.GetArrayFromImage(temp_img)
        temp_max = np.amax(temp_nda)
        if temp_max > modality_maximum:
            modality_maximum = temp_max
            print modality_maximum
    return modality_maximum

def ModalityMinimum(filepathes):
    ''' This code returns the minimum value for MR images'''
    modality_minimum = 4000
    for i in range(len(filepathes)):
        temp_img = sitk.ReadImage(filepathes[i])
        temp_nda = sitk.GetArrayFromImage(temp_img)
        temp_max = np.amin(temp_nda)
        if temp_max < modality_minimum: 
            modality_minimum = temp_max
            print modality_minimum
    return modality_minimum

#def Brats2017LesionsPathes(bratsPath):
#	''' This function gives you the pathes for evey lesion files'''
#	necrosis_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
#	for name in files if 'MNI' not in name and 'necrosis' in name and name.endswith('.nii.gz')]
#	necrosis_pathes.sort()
#	edema_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
#	for name in files if 'MNI' not in name and 'edema' in name and name.endswith('.nii.gz')]
#	edema_pathes.sort()
#	enhancing_tumor_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
#	for name in files if 'MNI' not in name and 'enhancingTumor' in name and name.endswith('.nii.gz')]
#	enhancing_tumor_pathes.sort()
#	return necrosis_pathes, edema_pathes, enhancing_tumor_pathes

#def Brats2017LesionsMNI152Pathes(bratsPath):
#	''' This function gives you the pathes for evey lesion in MNI 152 space files'''
#	necrosis_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
#	for name in files if 'MNI' in name and 'necrosis' in name and name.endswith('.nii.gz')]
#	necrosis_pathes.sort()
#	edema_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
#	for name in files if 'MNI' in name and 'edema' in name and name.endswith('.nii.gz')]
#	edema_pathes.sort()
#	enhancing_tumor_pathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
#	for name in files if 'MNI' in name and 'enhancingTumor' in name and name.endswith('.nii.gz')]
#	enhancing_tumor_pathes.sort()
#	return necrosis_pathes, edema_pathes, enhancing_tumor_pathes

def Brats2018FilePaths(bratsPath):
	''' This fucntion gives the filepathes of all MR images with N4ITK and ground truth'''
	t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
					for name in files if 't1' in name and 'ce' not in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1_filepaths.sort()

	t1c_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
					for name in files if 't1' in name and 'ce' in name and 'corrected' in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1c_filepaths.sort()

	t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
					for name in files if 't2' in name and 'corrected' in name and 'normalized' not in name 
					and 'MNI152' not in name and name.endswith('.nii.gz')]
	t2_filepaths.sort()

	flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
						for name in files if 'flair' in name and 'corrected' in name and 'normalized' not in name 
						and 'MNI152' not in name and name.endswith('.nii.gz')]
	flair_filepaths.sort()

	seg_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
					for name in files if 'seg' in name and 'FAST' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	seg_filepaths.sort()

	assert (len(t1_filepathes) == len(t1c_filepathes) == len(t2_filepathes) == len(flair_filepathes) == len(seg_filepathes)), "The len of different image modalities are differnt!!!"

	return t1_filepaths, t1c_filepaths, t2_filepaths, flair_filepaths, seg_filepaths

def Brats2018OriginalFilePaths(bratsPath):
	''' This fucntion gives the filepathes of all original MR images and ground truth'''
	t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
					for name in files if 't1' in name and 'ce' not in name and 'corrected' not in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1_filepaths.sort()

	t1c_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) 
					for name in files if 't1' in name and 'ce' in name and 'corrected' not in name 
					and 'normalized' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	t1c_filepaths.sort()

	t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
					for name in files if 't2' in name and 'corrected' not in name and 'normalized' not in name 
					and 'MNI152' not in name and name.endswith('.nii.gz')]
	t2_filepaths.sort()

	flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
						for name in files if 'flair' in name and 'corrected' not in name and 'normalized' not in name 
						and 'MNI152' not in name and name.endswith('.nii.gz')]
	flair_filepaths.sort()

	seg_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
					for name in files if 'seg' in name and 'FAST' not in name and 'MNI152' not in name and name.endswith('.nii.gz')]
	seg_filepaths.sort()

	assert len(t1_filepaths)==len(t1c_filepaths)==len(t2_filepaths)==len(flair_filepaths)==len(seg_filepaths), "Lengths are different!!!"

	return t1_filepaths, t1c_filepaths, t2_filepaths, flair_filepaths, seg_filepaths


def BrainMaskPaths(bratsPath):
	''' This function gives you the location of brain mask'''
	brain_mask_paths = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath) for name in files if 'brainmask' in name and name.endswith('.nii.gz')]
	brain_mask_paths.sort()
	return brain_mask_paths

def FindOneElement(s, ch):
	''' This function gives the indexs of one element ch on the string s'''
	return [i for i, ltr in enumerate(s) if ltr == ch]

def SubjectID(bratsPath):
	''' This function gives you the subject ID'''
	#filepathes = [os.path.join(root, name) for root, dirs, files in os.walk(bratsPath)
	#				for name in files if 't1' in name and 'ce' not in name and 'corrected' in name 
	#				and name.endswith('.nii.gz')]
	#filepathes.sort()
	#subjectID = [name[FindOneElement(name,'/')[-2]+1:FindOneElement(name,'/')[-1]] for name in filepathes]
	
	#return subjectID
	return bratsPath[FindOneElement(bratsPath,'/')[6]+1:FindOneElement(bratsPath,'/')[7]]

def AllSubjectID(bratsPath):
	''' This function gives you all subject IDs'''
	_, _, _, _, seg_filepathes = Brats2017FilePathes(bratsPath)
	subject_dirs = [os.path.split(seg_path)[0] for seg_path in seg_filepathes]
	all_subject_ID = [SubjectID(seg_path) for seg_path in seg_filepathes]
	return all_subject_ID


