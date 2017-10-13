# -*- coding: utf-8 -*-

import xml.etree.cElementTree as ET

def char_level_preprocess():
	# store each sentence and its corresponding label
	exampleList = []
	labelList = []
	save_to = 'weibo_sentence_char.txt'

	tree = ET.ElementTree(file='微博情绪标注语料.xml')
	root = tree.getroot()
	# print 'root.tag:', root.tag, 'root.attrib:', root.attrib

	count = 0
	for sentence in tree.iter(tag='sentence'):
		# some data has null 'emotion-1-type'
		if 'emotion-1-type' not in sentence.attrib or sentence.attrib['emotion-1-type'] == '':
			continue
		count += 1
		emotion_1_type = sentence.attrib['emotion-1-type']
		# utf-8 str compare test
		# print type(emotion_1_type), emotion_1_type.encode('utf-8') 
		text = sentence.text # unicode
		text_char_with_space = " ".join([char.encode('utf-8') for char in list(text)])
		exampleList.append(text_char_with_space)
		labelList.append(emotion_1_type)

	with open(save_to, 'w') as f:
		for example, label in zip(exampleList, labelList):
			f.write(example)
			f.write(' | ')
			f.write(label.encode('utf-8'))
			f.write('\n')

	print 'Example count:', count

def word_level_preprocess():
	pass

if __name__ == '__main__':
	char_level_preprocess()