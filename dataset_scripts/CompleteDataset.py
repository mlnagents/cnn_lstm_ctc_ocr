from shutil import copyfile
import os
import sys
# from pudb import set_trace; set_trace()

dir_dataset = '/Users/kalinin/Desktop/Dataset_add/MJSynth/'
dir_txt = '/Users/kalinin/Desktop/Dataset_add/MJSynth.txt'
dir_img = '/Users/kalinin/Desktop/mnt/ramdisk/max/90kDICT32px'
dir_symbols_frequency = '/Users/kalinin/Desktop/Dataset_symbols_new.txt'

need_these_symbols = {}

with open(dir_symbols_frequency, 'r') as file:
	content = file.readlines()

for line in content:
	line = line.replace('\n', '')
	symbol = line[0]
	frequency = int(line[line.find(' ')+1 : ])
	if symbol in 'abcdefghijklmnopqrstuvwxyz':
		need_these_symbols[symbol] = frequency

def text_checker(text_to_check):
	enough_symbols = False
	these_symbols = ''
	not_these_symbols = ''
	for i in need_these_symbols.items():
		if i[1] < 5000: # будут добавляться изображения, текст которых содержит следующие символы
			these_symbols += i[0]
		elif i[1] > 18000: # изображения с символами, которые встречались столько раз, добавляться в датасет не будут
			not_these_symbols += i[0]
	if len(these_symbols) == 0:
		enough_symbols = True

	text_ok = False
	for i in text_to_check.lower():
		if i in these_symbols:
			text_ok = True
		if i in not_these_symbols or i not in 'abcdefghijklmnopqrstuvwxyz':
			text_ok = False
			break

	if text_ok == True:
		for i in text_to_check.lower():
			try:
				need_these_symbols[i] += 1
			except:
				pass

	return text_ok, enough_symbols

count_all = 0
count = 0
with open(dir_txt, 'w') as f:
	for root, dirs, files in os.walk(dir_img):
		for file_name in files:
			if file_name[-3:] == 'jpg':
				count_all += 1
				text = file_name.split('_',2)[1]
				text_ok, enough_symbols = text_checker(text)
				if text_ok == True:
					new_dir = dir_dataset + file_name
					img_path = os.path.join(root, file_name)
					if os.path.exists(new_dir):
						# print('файл уже существует! запись под новым именем', new_dir)
						file_name = '(' + str(count) + ')' + file_name
						new_dir = dir_dataset + file_name
					copyfile(img_path, new_dir)
					f.write(new_dir + ' ' + text + '\n')
					count += 1
				if enough_symbols == True:
						print('Done')
						print(need_these_symbols)
						sys.exit(count)
		print(count_all, count)
print('Not enough images')
print(need_these_symbols)
