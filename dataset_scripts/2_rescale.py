from PIL import Image
import PIL
import os
from shutil import copyfile

dir_dataset = '/Users/kalinin/Desktop/Dataset/'
dir_txt = ['BornDigitalImages.txt', 'COCO.txt', 'FocusedSceneText.txt', 'Sport.txt', 'MA_packs.txt',
		   'IIIT5K.txt', 'IncidentalSceneText.txt', 'MJSynth.txt', 'MLT.txt', 'Medicine.txt']

count = 0
count_rotate = 0
ratio_mean = 0

for dataset in dir_txt:
	dir_dataset_txt = dir_dataset + dataset
	print(dataset, '...')
	with open(dir_dataset_txt, 'r') as t:
		content = t.readlines()
	
	for line in content:
		count += 1
		line = line.strip()
		file_dir = line[ : line.find(' ')]
		file_name = file_dir[file_dir.rfind('/')+1 : ]
		file_text = line[line.find(' ')+1 : ]

		file_dir_path = dir_dataset[:-1] + file_dir
		im = Image.open(file_dir_path)
		width, height = im.size
		height_new = 32
		width_new = int(width*(height_new/height))

		os.remove(file_dir_path)
		im = im.resize((width_new, height_new), PIL.Image.ANTIALIAS)
		im.save(file_dir_path)
		im.close()

print(count)