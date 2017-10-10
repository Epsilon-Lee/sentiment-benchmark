import torch
import operator

specials = {'<pad>':0, '<unk>': 1}

class Dictionary(object):

	def __init__(self, sentenceList, savedDataFile=None):

		if savedDataFile != None:
			savedDict = torch.load(savedDataFile)['dict']
			self.idx2label = savedDict['idx2label']
			self.label2idx = savedDict['label2idx']
			self.freq_lst = savedDict['freq_dict']
		else:
			self.idx2label = {}
			self.label2idx = {}
			self.computeWordFreq(sentenceList)

	def computeWordFreq(self, sentenceList):
		freq_dict = {}
		for sentence in sentenceList:
			words = sentence.split()
			for word in words:
				if word in freq_dict:
					freq_dict[word] += 1
				else:
					freq_dict[word] = 0
		# sort by frequency with descending order
		self.freq_lst = sorted(freq_dict, key=operator.itemgetter(1), reverse=True)

	# this method rebuild idx2label and label2idx dicts
	def prune(self, vocab_size):
		if specials != None:
			for label, idx in specials.iteritems():
				self.idx2label[idx] = label
				self.label2idx[label] = idx

		idx = len(self.idx2label)
		for i in range(vocab_size):
			self.idx2label[idx] = self.freq_lst[i][0]
			self.label2idx[self.freq_lst[i][0]] = idx
			idx += 1

	def convertToLabel(self, idx_seq):
		label_seq = []
		for idx in idx_seq:
			label_seq.append(self.idx2label[idx])
		return label_seq

	def converToIdx(self, label_seq):
		idx_seq = []
		for label in label_seq:
			if label not in self.label2idx:
				idx_seq.append(self.label2idx['<unk>'])
			else:
				idx_seq.append(self.label2idx[label])
		return idx_seq