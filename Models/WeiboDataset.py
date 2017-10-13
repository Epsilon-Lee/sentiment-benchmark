# -*- coding: utf-8 -*-

import torch
import argparse
import math
import random

import Dictionary

parser = argparse.ArgumentParser("preprocess opts")
parser.add_argument("-weiboFilePath", type=str)
parser.add_argument("-vocab_size", type=int)

args = parser.parse_args()

class WeiboDataset(object):

	def __init__(self, examples_idx, labels_idx, batch_size):
		self.examples_idx = examples_idx
		self.labels_idx = labels_idx
		self.batch_size = batch_size
		self.batch_num = math.ceil(len(examples_idx) / batch_size) # ceil(5/2) = 2 + 1 = 3

	def _batchify(self, batch_examples_idx, batch_labels_idx):
		batch_lengths = []
		for example_idx in batch_examples_idx:
			batch_lengths.append(len(example_idx))
		max_len = max(batch_lengths)

		# create a torch.LongTensor matrix to store a batch of idx for each example
	 	## initialize to fill with 0
	 	batch_tensor = torch.LongTensor(len(batch_lengths), max_len).fill_(Dictionary.specials['<pad>'])
	 	mask_tensor = torch.FloatTensor(len(batch_lengths), max_len).fill_(0)
	 	for i, example_idx in enumerate(batch_examples_idx):
	 		example_tensor = torch.LongTensor(example_idx)
	 		batch_tensor[i].narrow(0, 0, batch_lengths[i]).copy_(example_tensor) # tensor returned by narrow share storage with original tensor
	 		mask_tensor[i].narrow(0, 0, batch_lengths[i]).copy_(torch.ones(batch_lengths[i]))

	 	label_tensor = torch.LongTensor(batch_labels_idx)

	 	# batch_tensor shape: (batch_size x max_len)
	 	# mask_tensor shape: (batch_size x max_len)
	 	# label_tensor shape: (batch_size)
	 	return batch_tensor, mask_tensor, label_tensor

	def __getitem__(self, index):
		# index should between [0, batch_num - 1], otherwise no data could be accessed
		assert index < self.batch_num, "%d > %d" % (index, self.batch_num)
		batch_examples_idx = self.examples_idx[index * self.batch_size : (index + 1) * self.batch_size]
		batch_labels_idx = self.labels_idx[index * self.batch_size : (index + 1) * self.batch_size]
		batch_tensor, label_tensor, mask_tensor = self._batchify(batch_examples_idx, batch_labels_idx)
		return batch_tensor, label_tensor, mask_tensor
		
	def __len__(self):
		return self.batch_num

	def shuffle(self):
		examples_and_labels = zip(self.examples_idx, self.labels_idx)
		random.shuffle(examples_and_labels)
		self.examples_idx, self.labels_idx = zip(*examples_and_labels)

	def saveDataset(self, path):
		saveDataset = {'examples_idx' : self.examples_idx,
						'labels_idx' : self.labels_idx,
						'batch_size' : self.batch_size,
						'batch_num' : self.batch_num}

	@classmethod
	def preprocess(cls, examples, labels, dictionary):
		
		# string clean method
		def clean_str(string):
			return string

		# check len(examples) == len(labels)
		if len(examples) != len(labels):
			print 'examples labels mismatch'
			return None, None

		# build dictionary
		examples_idx = []
		labels_idx = []
		for example, label in zip(examples, labels):
			example_idx = dictionary.convertSymbolToIdx(example.split())
			examples_idx.append(example_idx)
			label_idx = dictionary.convertLabelToIdx(label)
			labels_idx.append(label_idx)

		return examples_idx, labels_idx