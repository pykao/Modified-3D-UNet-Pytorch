import argparse
from dataset import run_preprocessing_BraTS2018_training #, run_preprocessing_BraTS2018_validationOrTesting
import paths
print 'start'
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="train for training set, val for validation set, and test for testing set", type=str)
args = parser.parse_args()
print args.mode

if args.mode == "train":
    run_preprocessing_BraTS2018_training(paths.raw_training_data_folder, paths.preprocessed_training_data_folder)
#elif args.mode == "val":
#    run_preprocessing_BraTS2017_trainSet(paths.raw_validation_data_folder, paths.preprocessed_validation_data_folder)
#elif args.mode == "test":
#	run_preprocessing_BraTS2017_trainSet(paths.raw_testing_data_folder, paths.preprocessed_testing_data_folder)
else:
	raise ValueError("Unknown value for --mode. Use \"train\", \"test\" or \"val\"")