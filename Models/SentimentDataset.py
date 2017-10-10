import torch
import argparse
import math

import Dictionary

parser = argparse.ArgumentParser("preprocess opts")
parser.add_argument("-filePos", type=str)
parser.add_argument("-fileNeg", type=str)
parser.add_argument("-vocab_size", type=int)

args = parser.parse_args()

class SentimentDataset(object):

	def __init__(self, examples, labels, batch_size):
		self.examples = examples
		self.labels = labels
		self.batch_size = batch_size
		self.batch_num = math.ceil(len(examples) / batch_size)

	def __getitem__(self, index):
		# index should be less than self.batch_num, otherwise no data could be accessed
		assert index < self.batch_num, "%d > %d" % (index, self.batch_num)
		

	@classmethod
	def preprocess(cls):
		examples = [] # save sentence's word idx sequence
		labels = [] # save corresponding label: 0 for neg, 1 for pos
		
		# string clean method
		def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

		# readfile to examples
		## read filePos, label add '1's
		with open(args.filePos, 'r') as f:
			lines = f.readlines()
			for str_i in lines:
				str_i = clean_str(str_i)
				examples += [str_i]
				labels += [1]
		## read fileNeg, label add '0's
		with open(args.fileNeg, 'r') as f:
			lines = f.readlines()
			for str_i in lines:
				str_i = clean_str(str_i)
				examples += [str_i]
				labels += [0]

		# check len(examples) == len(labels)
		if len(examples) != len(labels):
			print 'examples labels mismatch'
			return None, None

		# build dictionary
		dictionary = Dictionary(examples)
		dictionary.prune(args.vocab_size)
		examples_idx = []
		for example in examples:
			example_idx = dictionary.convertToIdx(examples.split())
			examples_idx += [example_idx]

		return examples_idx, labels