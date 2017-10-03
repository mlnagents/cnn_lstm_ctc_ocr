import os
import sys
import operator

dir_dataset = '/Users/kalinin/Desktop/Dataset'
dir_txt = ['BornDigitalImages.txt', 'COCO.txt', 'FocusedSceneText.txt', 'Sport.txt', 'MA_packs.txt',
		   'IIIT5K.txt', 'IncidentalSceneText.txt', 'MJSynth.txt', 'MLT.txt', 'Medicine.txt']
dir_alphabet = '/Users/kalinin/Desktop/Dataset/Dataset_symbols.txt'
dir_new_alphabet = '/Users/kalinin/Desktop/Dataset/Dataset_symbols_new.txt'

count = 0
count_not_skip = 0
new_alphabet = {}
old_alphabet = {}

with open(dir_alphabet, 'r') as file:
	content = file.readlines()

for line in content:
	line = line.replace('\n', '')
	old_symbol = line[0]
	new_symbol = line[line.rfind(' ')+1 : ]
	old_alphabet[old_symbol] = new_symbol

for dataset in dir_txt:
	dir_dataset_txt = os.path.join(dir_dataset, dataset)
	print(dataset, '...')
	with open(dir_dataset_txt, 'r') as t:
		content = t.readlines()

	dataset_new = dataset[ : dataset.rfind('.')]
	dataset_new = dataset_new + '_new.txt'
	dir_dataset_new_txt = os.path.join(dir_dataset, dataset_new)
	with open(dir_dataset_new_txt, 'w') as f:	
		for line in content:
			line = line.strip()
			file_dir = line[ : line.find(' ')]
			file_name = file_dir[file_dir.rfind('/')+1 : ]
			file_text = line[line.find(' ')+1 : ]
			count += 1

			skip_image = False
			file_text_new = ''
			for symbol in file_text:
				skip_symbol = False
				replace_symbol = old_alphabet[symbol]
				if replace_symbol == 'OK':
					symbol_new = symbol
				elif replace_symbol == 'DEL':
					skip_image = True
					break
				elif replace_symbol == 'SKIP':
					symbol_new = ''
					skip_symbol = True
				elif replace_symbol == 'NEW_SYMBOL':
					sys.exit('не должно быть NEW_SYMBOL, выход из программы')
				else:
					symbol_new = replace_symbol

				if skip_symbol == False:
					try:
						new_alphabet[symbol_new] += 1
					except:
						new_alphabet[symbol_new] = 1

				file_text_new += symbol_new

			if skip_image == False:
				f.write(file_dir + ' ' + file_text_new + '\n')
				count_not_skip += 1


sorted_new_alphabet = sorted(new_alphabet.items(), key=operator.itemgetter(1))

with open(dir_new_alphabet, 'w') as f:
	for i in sorted_new_alphabet:
		f.write(str(i[0]) + ' ' + str(i[1]) + '\n')

print('Done', 'всего изображений:', count, 'пропущено:', (count - count_not_skip))
