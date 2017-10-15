# -*- coding: utf-8 -*-

import torch
import operator

specials = {'<pad>':0, '<unk>': 1}

class Dictionary(object):

	def __init__(self, sentenceList, labelList, savedDictFile=None):

		if savedDictFile != None:
			savedDict = torch.load(savedDictFile)
			
			self.idx2symbol = savedDict['idx2symbol']
			self.symbol2idx = savedDict['symbol2idx']
			self.symbol_freq_lst = savedDict['symbol_freq_lst']

			self.idx2label = savedDict['idx2label']
			self.label2idx = savedDict['label2idx']
			self.label_freq_lst = savedDict['label_freq_lst']
		else:
			# symbol2idx, idx2symbol: '生命' -> 54, 32 -> '青岛' 
			self.symbol2idx = {}
			self.idx2symbol = {}
			# idx2label, label2idx: 0 -> '伤心', '快乐' -> 1
			self.idx2label = {}
			self.label2idx = {}
			self.computeWordFreq(sentenceList)
			self.computeLabelFreq(labelList)

	def computeWordFreq(self, sentenceList):
		freq_dict = {}
		for sentence in sentenceList:
			words = sentence.split()
			for word in words:
				if word in freq_dict:
					freq_dict[word] += 1
				else:
					freq_dict[word] = 1
		
		# sort by frequency with descending order
		self.symbol_freq_lst = sorted(freq_dict.items(), key=lambda tup: tup[1], reverse=True)

	def computeLabelFreq(self, labelList):
		label_freq_dict = {}
		for label in labelList:
			if label not in label_freq_dict:
				label_freq_dict[label] = 1
			else:
				label_freq_dict[label] += 1

		self.label_freq_lst = sorted(label_freq_dict.items(), key=lambda tup: tup[1], reverse=True)

		# build label2idx/idx2label dicts
		new_label_index = 0
		for label, freq in self.label_freq_lst:
			self.label2idx[label] = new_label_index
			self.idx2label[new_label_index] = label
			new_label_index += 1

	# this method build/rebuild idx2label and label2idx dicts
	def prune_vocabsize(self, vocab_size):
		if specials != None:
			for symbol, idx in specials.iteritems():
				self.idx2symbol[idx] = symbol
				self.symbol2idx[symbol] = idx

		idx = len(self.idx2symbol)
		for i in range(vocab_size):
			self.idx2symbol[idx] = self.symbol_freq_lst[i][0]
			self.symbol2idx[self.symbol_freq_lst[i][0]] = idx
			idx += 1

	def prune_freq(self, cut_freq):
		if specials != None:
			for symbol, idx in specials.iteritems():
				self.idx2symbol[idx] = symbol
				self.symbol2idx[symbol] = idx

		idx = len(self.idx2symbol)
		for symbol, freq in self.symbol_freq_lst:
			if freq < cut_freq:
				break
			self.idx2symbol[idx] = symbol
			self.symbol2idx[symbol] = idx
			idx += 1

		print 'Pruned by frequency %d ... vocabulary size: %d' % (cut_freq, idx)

	def convertIdxToSymbol(self, idx_seq):
		symbol_seq = []
		for idx in idx_seq:
			symbol_seq.append(self.idx2symbol[idx])
		return symbol_seq

	def convertSymbolToIdx(self, symbol_seq):
		# convert out-of-vocabulary symbol to unk
		idx_seq = []
		for symbol in symbol_seq:
			if symbol not in self.symbol2idx:
				idx_seq.append(self.symbol2idx['<unk>'])
			else:
				idx_seq.append(self.symbol2idx[symbol])
		return idx_seq

	def convertLabelToIdx(self, label):
		return self.label2idx[label]

	def convertIdx2Label(self, idx):
		return self.idx2label[idx]

	def originDictSize(self):
		if self.symbol_freq_lst != None:
			return len(self.symbol_freq_lst)
		else:
			return -1

	def saveDict(self, path):
		saveDict = {'idx2symbol':self.idx2symbol,
			'symbol2idx':self.symbol2idx,
			'symbol_freq_lst':self.symbol_freq_lst,
			'idx2label':self.idx2label,
			'label2idx':self.label2idx,
			'label_freq_lst':self.label_freq_lst}

		torch.save(saveDict, path)
		