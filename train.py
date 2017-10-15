# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import argparse
import math

from Models.WeiboDataset import WeiboDataset
from Models.Dictionary import Dictionary
from Models.Models import EmbAvg, RNNHidAvg

parser = argparse.ArgumentParser(description="training options")

# data loading
parser.add_argument("-dataset_path", type=str, default="./NLPCC/dataset.pt")
parser.add_argument("-dictionary_path", type=str, default="./NLPCC/dict.pt")
parser.add_argument("-checkpoint_path", type=str, default="")

# neural network related
parser.add_argument("-vocab_size", type=int)
parser.add_argument("-emb_size", type=int, default=128)
parser.add_argument("-class_num", type=int, default=7)
parser.add_argument("-hid_size", type=int, default=128)
parser.add_argument("-rnn_type", type=str, default="lstm") # you can choose one of: ["rnn", "gru", "lstm"]

# optimizer
parser.add_argument("-optimizer_type", type=str, default='sgd')
parser.add_argument("-learning_rate", type=float, default=0.001)
parser.add_argument("-batch_size", type=int, default=32)

# GPU
parser.add_argument("-gpu_id", type=int, default=0)
parser.add_argument("-cuda", type=int, default=1)

# training and logging
parser.add_argument("-max_epoch", type=int, default=100)
parser.add_argument("-logging_interval", type=int, default=32)

opts = parser.parse_args()

if opts.cuda:
	print 'Using gpu', opts.gpu_id
	torch.cuda.set_device(opts.gpu_id)

def critierion():
	cross_entropy_loss = nn.CrossEntropyLoss()
	if opts.cuda:
		cross_entropy_loss.cuda()
	return cross_entropy_loss

def train(model, optimizer, dictionary, epoch_num, trainDataset, devDataset, testDataset):

	# inner function
	def trainEpoch(epoch_num, batch_count, loss_lst):
		model.train()
		trainDataset.shuffle()
		batch_loss = 0

		for idx in range(len(trainDataset)):
			# get one batch data
			batch_tensor, label_tensor, mask_tensor = trainDataset[idx]
			# print 'idx', idx, 'batch_tensor', batch_tensor.size(), 'mask_tensor', mask_tensor.size()
			if opts.cuda:
				batch_tensor = batch_tensor.cuda()
				label_tensor = label_tensor.cuda()
				mask_tensor = mask_tensor.cuda()
				model.cuda()

			batch_tensor_var = Variable(batch_tensor)
			label_tensor_var = Variable(label_tensor)
			mask_tensor_var = Variable(mask_tensor)

			model.zero_grad()
			probs = model(batch_tensor_var, mask_tensor_var)
			loss = cross_entropy_loss(probs, label_tensor_var)

			loss.backward()
			optimizer.step()

			# train accuracy per batch
			probs_data = probs.data
			max_probs, pred_classes = torch.max(probs_data, 1) # shapes both are batch_size
			acc = torch.eq(pred_classes, label_tensor).sum()

			example_loss = loss.data[0]
			batch_count += 1
			loss_lst.append((batch_count, example_loss))
			# logging every logging_interval batches
			if batch_count % opts.logging_interval == 0:
				print "Epoch", epoch_num, "Batches id", (idx + 1), "Batch count", batch_count, "Batch loss", example_loss, "Batch accuracy", acc * 1. / trainDataset.batch_size

	# 1. show info about model and dataset
	print 'Model architecture'
	print model

	print 'Dataset information'
	# print 'batch size:', trainDataset.batch_size
	print 'train batch_num:', len(trainDataset), 'batch size:', trainDataset.batch_size
	print 'dev   batch_num:', len(devDataset), 'batch size:', devDataset.batch_size
	print 'test  batch_num:', len(testDataset), 'batch size:', testDataset.batch_size

	print 'Dictionary information'
	print 'vocab_size:', len(dictionary.symbol2idx)

	# 2. trainEpoch (epoch means one-pass over the whole dataset)
	loss_lst = []
	batch_count = 0
	cross_entropy_loss = critierion()
	for epoch_idx in range(epoch_num, opts.max_epoch):
		trainEpoch(epoch_idx, batch_count, loss_lst)

		# 3. evaluate on dev and test
		model.eval()
		dev_loss = 0
		dev_acc_count = 0
		for idx in range(len(devDataset)):
			dev_batch_tensor, dev_label_tensor, dev_mask_tensor = devDataset[idx]
			if opts.cuda:
				dev_batch_tensor = dev_batch_tensor.cuda()
				dev_label_tensor = dev_label_tensor.cuda()
				dev_mask_tensor = dev_mask_tensor.cuda()
				model.cuda()
			dev_batch_tensor_var = Variable(dev_batch_tensor)
			dev_label_tensor_var = Variable(dev_label_tensor)
			dev_mask_tensor_var = Variable(dev_mask_tensor)
			dev_probs = model(dev_batch_tensor_var, dev_mask_tensor_var)
			dev_batch_loss = cross_entropy_loss(dev_probs, dev_label_tensor_var)
			dev_loss += dev_batch_loss.data[0]

			dev_probs_data = dev_probs.data
			max_probs, predict_classes = torch.max(dev_probs_data, 1)
			dev_acc_count += torch.eq(predict_classes, dev_label_tensor).sum()
		print 'Evaluation on devDataset'
		print 'Accuracy', dev_acc_count * 1. / len(devDataset.examples_idx), 'Loss per batch', dev_loss / devDataset.batch_num
		test_loss = 0
		test_acc_count = 0
		for idx in range(len(testDataset)):
			test_batch_tensor, test_label_tensor, test_mask_tensor = testDataset[idx]
			if opts.cuda:
				test_batch_tensor = test_batch_tensor.cuda()
				test_label_tensor = test_label_tensor.cuda()
				test_mask_tensor = test_mask_tensor.cuda()
				model.cuda()
			test_batch_tensor_var = Variable(test_batch_tensor)
			test_label_tensor_var = Variable(test_label_tensor)
			test_mask_tensor_var = Variable(test_mask_tensor)
			test_probs = model(test_batch_tensor_var, test_mask_tensor_var)
			test_batch_loss = cross_entropy_loss(test_probs, test_label_tensor_var)
			test_loss += test_batch_loss.data[0]

			test_probs_data = test_probs.data
			max_probs, predict_classes = torch.max(test_probs_data, 1)
			test_acc_count += torch.eq(predict_classes, test_label_tensor).sum()
		print 'Evaluation on testDataset'
		print 'Accuracy', test_acc_count * 1. / len(testDataset.examples_idx), 'Loss per batch', test_loss / testDataset.batch_num
		print 
		# 4. save checkpoint

def predict(model, devDataset, testDataset):
	# model.eval()
	pass

if __name__ == '__main__':
	
	# 1. load dataset and dictionary
	dataset = torch.load(opts.dataset_path)
	trainDataset = dataset["trainDataset"]
	devDataset = dataset["devDataset"]
	testDataset = dataset["testDataset"]
	trainDataset.batch_size = opts.batch_size
	trainDataset.batch_num = math.ceil(len(trainDataset.examples_idx)/opts.batch_size)

	# savedDict = torch.load(opts.dictionary_path)
	dictionary = Dictionary(None, None, opts.dictionary_path)
	opts.vocab_size = len(dictionary.symbol2idx)

	# 2. create model or load from checkpoint
	# EmbAvg model
	# model = EmbAvg(opts)
	# RNNHidAvg model
	model = RNNHidAvg(opts)

	if opts.checkpoint_path != "":
		print "Loading checkpoint to initialize the model"
		# TO-DO

		checkpoint = torch.load(opts.checkpoint_path)
		model_state_dict = checkpoint['model_state_dict']
		model.load_state_dict(model_state_dict)
		optimizer = checkpoint['optimizer']
		epoch_num = checkpoint['epoch_num']
	else:
		# 3. create optimizer
		epoch_num = 1
		if opts.optimizer_type == "sgd":
			optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate)
		elif opts.optimizer_type == "adam":
			opts.optimizer_type == optim.Adam(model.parameters, lr=opts.learning_rate)

	# 4. start training
	train(model, optimizer, dictionary, epoch_num, trainDataset, devDataset, testDataset)