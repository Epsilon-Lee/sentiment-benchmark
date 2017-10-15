import torch
import torch.nn as nn
import torch.nn.functional as F

# EmbAvg model
class EmbAvg(nn.Module):
	def __init__(self, opts):
		super(EmbAvg, self).__init__()

		self.emb = nn.Embedding(opts.vocab_size, opts.emb_size)
		self.linear = nn.Linear(opts.emb_size, opts.class_num)

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
class RNNHidAvg(nn.Module):
	def __init__(self, opts):
		super(RNNHidAvg, self).__init__()
		self.emb = nn.Embedding(opts.vocab_size, opts.emb_size)
		# 3 types of rnn
		if opts.rnn_type == 'rnn':
			self.rnn = nn.RNN(opts.emb_size, opts.hid_size)
			self.rnn_type = 'rnn'
		elif opts.rnn_type == 'lstm':
			self.rnn = nn.LSTM(opts.emb_size, opts.hid_size)
			self.rnn_type = 'lstm'
		else:
			self.rnn = nn.GRU(opts.emb_size, opts.hid_size)
			self.rnn_type = 'gru'
		self.linear = nn.Linear(opts.hid_size, opts.class_num)

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

# unit test
if __name__ == '__main__':
	
	class Options:
		emb_size = 128
		hid_size = 128
		vocab_size = 1000
		class_num = 7
		rnn_type = 'lstm'

	opts = Options()
	rnnHidAvg = RNNHidAvg(opts)
	print rnnHidAvg
