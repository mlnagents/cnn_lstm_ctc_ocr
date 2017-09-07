from PIL import Image
import os
import sys

dir_dataset = '/Users/kalinin/Desktop/Dataset'
dir_txt = ['BornDigitalImages.txt', 'COCO.txt', 'FocusedSceneText.txt', 'Sport.txt', 'MA_packs.txt',
		   'IIIT5K.txt', 'IncidentalSceneText.txt', 'MJSynth.txt', 'MLT.txt', 'Medicine.txt']

count_png = 0
count_jpg = 0

for dataset in dir_txt:
	dir_dataset_txt = os.path.join(dir_dataset, dataset)
	print(dataset, '...')
	with open(dir_dataset_txt, 'r') as t:
		content = t.readlines()
	
	os.remove(dir_dataset_txt)

	with open(dir_dataset_txt, 'w') as t:
		for line in content:
			line = line.strip()
			file_dir = line[ : line.find(' ')]
			file_name = file_dir[file_dir.rfind('/')+1 : ]
			file_text = line[line.find(' ')+1 : ]

			if file_dir[-4:] == '.png':
				count_png += 1
				im = Image.open(dir_dataset + file_dir)
				new_dir = dir_dataset + file_dir[:-3] + 'jpg'
				dir_txt = file_dir[:-3] + 'jpg'
				if os.path.exists(new_dir):
					sys.exit('WTF', file_dir)
				im.save(new_dir, "JPEG")
				im.close()
				os.remove(dir_dataset + file_dir)
			else:
				count_jpg += 1
				dir_txt = file_dir

			t.write(dir_txt + ' ' + file_text + '\n')

print(count_png)
print(count_jpg)