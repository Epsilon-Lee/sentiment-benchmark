# -*- coding: utf-8 -*-

from Models.WeiboDataset import WeiboDataset
from Models.Dictionary import Dictionary
import random
import torch

# 1. split train, dev, test
filePath = 'raw_data/weibo_sentence_char.txt'
trainPath = 'NLPCC/char_train.txt'
devPath = 'NLPCC/char_dev.txt'
testPath = 'NLPCC/char_test.txt'

train = []
dev = []
test = []
# with open(filePath, 'r') as f:
# 	lines = f.readlines()
# 	example_num = len(lines)
# 	# split 
# 	# - dev 1000; 
# 	# - test 1000; 
# 	# - rest train. 
# 	random.shuffle(lines)

# 	# list indexing
# 	dev = lines[:1000]
# 	test = lines[1000:2000]
# 	train = lines[2000:]

# 	with open(trainPath, 'w') as trainFile:
# 		for line in train:
# 			trainFile.write(line)

# 	with open(devPath, 'w') as devFile:
# 		for line in dev:
# 			devFile.write(line)

# 	with open(testPath, 'w') as testFile:
# 		for line in test:
# 			testFile.write(line)

with open(trainPath, 'r') as trainFile:
	train = trainFile.readlines()
with open(devPath, 'r') as devFile:
	dev = devFile.readlines()
with open(testPath, 'r') as testFile:
	test = testFile.readlines()

# 2. create dictionary
trainSentenceList = []
trainLabelList = []
for example in train:
	example_and_label = example.strip().split('|')
	example_merge = "|".join(example_and_label[:-1]).strip()
	label = example_and_label[-1].strip()
	trainSentenceList.append(example_merge)
	trainLabelList.append(label)
	# bug detection: some data have '' as label
	if label == '':
		print 'example_and_label:', example_and_label
		print 'example:', example.strip()

dictionary = Dictionary(trainSentenceList, trainLabelList)
print 'Dictionary size:', len(dictionary.symbol_freq_lst)
for k, v in dictionary.label_freq_lst:
	print k, v
''' train set statistics: 7 labels
喜好 2312
高兴 1724
厌恶 1695
悲伤 1284
愤怒 929
惊讶 381
恐惧 154
'''
cut_freq = 3
dictionary.prune_freq(cut_freq)
# print to have a direct feeling of the un-pruned dict
# sorted_symbol2idx = sorted(dictionary.symbol2idx.items(), key=lambda tup: tup[1])
# for symbol, idx in sorted_symbol2idx:
# 	print symbol, idx
print 'Original dictionary size:', dictionary.originDictSize()
dictionary.saveDict('./NLPCC/dict.pt')

# 3. create dataset and save as dataset.pt
devSentenceList = []
devLabelList = []
for example in dev:
	example_and_label = example.strip().split('|')
	example_merge = "|".join(example_and_label[:-1]).strip()
	label = example_and_label[-1].strip()
	devSentenceList.append(example_merge)
	devLabelList.append(label)

testSentenceList = []
testLabelList = []
for example in test:
	example_and_label = example.strip().split('|')
	example_merge = "|".join(example_and_label[:-1]).strip()
	label = example_and_label[-1].strip()
	testSentenceList.append(example_merge)
	testLabelList.append(label)

train_examples_idx, train_labels_idx = WeiboDataset.preprocess(trainSentenceList, trainLabelList, dictionary)
dev_examples_idx, dev_labels_idx = WeiboDataset.preprocess(devSentenceList, devLabelList, dictionary)
test_examples_idx, test_labels_idx = WeiboDataset.preprocess(testSentenceList, testLabelList, dictionary)

batch_size = 8
trainDataset = WeiboDataset(train_examples_idx, train_labels_idx, batch_size)
devDataset = WeiboDataset(dev_examples_idx, dev_labels_idx, batch_size)
testDataset = WeiboDataset(test_examples_idx, test_labels_idx, batch_size)

## save dataset
dataset = {'trainDataset':trainDataset, 'devDataset':devDataset, 'testDataset':testDataset}
torch.save(dataset, './NLPCC/dataset.pt')

# print batch_tensor
for idx in range(len(devDataset)):
	batch_tensor, _, _ =  devDataset[idx]
	print batch_tensor
	break