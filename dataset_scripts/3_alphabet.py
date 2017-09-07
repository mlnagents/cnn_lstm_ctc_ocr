import os
import sys
import operator
#from pudb import set_trace; set_trace()

dir_dataset = '/Users/kalinin/Desktop/Dataset'
dir_txt = ['BornDigitalImages.txt', 'COCO.txt', 'FocusedSceneText.txt', 'Sport.txt', 'MA_packs.txt',
		   'IIIT5K.txt', 'IncidentalSceneText.txt', 'MJSynth.txt', 'MLT.txt', 'Medicine.txt']
dir_alphabet = '/Users/kalinin/Desktop/Dataset/Dataset_symbols.txt'

count = 0
dict_alphabet = {}

with open(dir_alphabet, 'r') as f:
	content = f.readlines()

for line in content:
	line = line.replace('\n', '')
	old_symbol = line[0]
	new_symbol = line[line.rfind(' ')+1 : ]
	dict_alphabet[old_symbol] = [0, new_symbol]

for dataset in dir_txt:
	dir_dataset_txt = os.path.join(dir_dataset, dataset)
	print(dataset, '...')
	with open(dir_dataset_txt, 'r') as t:
		content = t.readlines()
	
	for line in content:
		count += 1
		line = line.strip()
		file_dir = line[ : line.find(' ')]
		file_name = file_dir[file_dir.rfind('/')+1 : ]
		file_text = line[line.find(' ')+1 : ]

		for symbol in file_text:
			try:
				value = dict_alphabet[symbol]
				value[0] = value[0] + 1
				dict_alphabet[symbol] = value
			except:
				dict_alphabet[symbol] = [1, 'NEW_SYMBOL']

sorted_dict_alphabet = sorted(dict_alphabet.items(), key=operator.itemgetter(1))

with open(dir_alphabet, 'w') as f:
	for i in sorted_dict_alphabet:
 		f.write(str(i[0]) + ' ' + str(i[1][0]) + ' ' + str(i[1][1]) + '\n')

print('Done', count)