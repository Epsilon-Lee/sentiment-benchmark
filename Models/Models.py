import torch
import torch.nn as nn
import torch.nn.functional as F

# EmbAvg model
class EmbAvg(nn.module):
	def __init__(self, opt):
		self.emb = nn.Embedding(opt.vocab_size, opt.emb_size)
		self.linear = nn.Linear(opt.emb_size, opt.class_num)

	def forward(self, input, mask):
	# input should be one-hot tensor Variable with shape: (batch_size x seq_len x vocab_size)
	# mask should have size: (batch_size x seq_len)
		word_embs = self.emb(input) # shape: (batch_size x seq_len x emb_size)
		masked_word_embs = word_embs * mask.unsqueeze(2) # shape: (batch_size x seq_len x emb_size)
		sent_embs = masked_word_embs.sum(1) # shape: (batch_size x emb_size)
		counts = mask.sum(1).unsqueeze(1) # shape: (batch_size x 1)
		avg_sent_embs = sent_embs / counts # shape: (batch_size x emb_size)

		unnormalized_predict = self.linear(avg_sent_embs) # shape: (batch_size x class_num)
		predict = F.softmax(unnormalized_predict)

		return predict # shape: (batch_size x class_num)

# RNNHidAvg model
class RNNHidAvg(nn.module):
	def __init__(self, opt):
		self.emb = nn.Embedding(opt.vocab_size, opt.emb_size)
		# 3 types of rnn
		if opt.rnn_type == 'rnn':
			self.rnn = nn.RNN(opt.emb_size, opt.hid_size)
			self.rnn_type = 'rnn'
		elif opt.rnn_type == 'lstm':
			self.rnn = nn.LSTM(opt.emb_size, opt.hid_size)
			self.rnn_type = 'lstm'
		else:
			self.rnn = nn.GRU(opt.emb_size, opt.hid_size)
			self.rnn_type = 'gru'
		self.linear = nn.Linear(opt.hid_size, opt.class_num)

	def forward(self, input, mask):
		# input shape: (batch_size x seq_len x emb_size)
		input_reshaped = input.transpose(0, 1) # shape: (seq_len x batch_size)
		# mask shape: (batch_size x seq_len)
		mask_reshaped = mask.transpose(0, 1) # shape: (seq_len x batch_size)
		word_embs = self.emb(input_reshaped) # shape: (seq_len x batch_size x emb_size)
		hids, _ = self.rnn(word_embs, None) # shape: (seq_len x batch_size x hid_size)
		masked_hids = hids * mask_reshaped.unsqueeze(2)
		sent_embs = masked_hids.sum(0) # shape: (batch_size x hid_size)
		unnormalized_predict = self.linear(sent_embs) # shape: (batch_size x class_num)
		predict = F.softmax(unnormalized_predict)

		return predict # shape: (batch_size x class_num)
