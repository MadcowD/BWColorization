"""
colorization.py
	Author: William H. Guss

	An implementation of colorful image colorization form ECCV 2016.
"""
import tensorflow as tf
import numpy as numpy
import argparse
import colorization.model

def parse_cmd_args():
	parser = argparse.ArgumentParser()
	main_subparsers = parser.add_subparsers(title="mode", dest="mode_command")

	# Sets up the parsers.
	train_parser = main_subparsers.add("train", help="Trains a BW-Colorization model.")
	train_parser.add_argument('imagenet_dir', type=str)

	test_parser = main_subparsers.add("test", help="Tests a BW-Colorization model.")
	test_parser.add_argument('model_path', type=str)

	args = parser.parse_args()

	return parser, args



def main():
	parser, args = parse_cmd_args()

