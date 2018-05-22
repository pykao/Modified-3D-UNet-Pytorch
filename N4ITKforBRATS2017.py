#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 18:54:21 2018

@author: pkao

This code applies N4ITK for BRATS2018 database

The output for this code should be name_of_mri_corrected.nii.gz
"""
import os
from nipype.interfaces.ants import N4BiasFieldCorrection
from multiprocessing import Pool

def N4ITK(filepath):
    print 'Working on: '+filepath 
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = filepath
    
    outputPath = filepath[:-7]+'_N4ITK_corrected.nii.gz'
    n4.inputs.output_image = outputPath
    
    n4.run()

brats2017_training_path = '/media/pkao/Dataset/BraTS2018/training'

t1_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2017_training_path)
for name in files if 't1' in name and 'ce' not in name and name.endswith('.nii.gz')]
t1_filepaths.sort()

t1ce_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2017_training_path)
for name in files if 't1ce' in name and name.endswith('.nii.gz')]
t1ce_filepaths.sort()

t2_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2017_training_path)
for name in files if 't2' in name and name.endswith('.nii.gz')]
t2_filepaths.sort()

flair_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brats2017_training_path)
for name in files if 'flair' in name and name.endswith('.nii.gz')]
flair_filepaths.sort()


file_paths = t1_filepaths + t1ce_filepaths + t2_filepaths + flair_filepaths

pool = Pool(6)

pool.map(N4ITK, file_paths)

