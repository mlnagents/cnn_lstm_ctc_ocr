from PIL import Image
import os
from shutil import copyfile

dir_dataset = '/Users/kalinin/Desktop/Dataset/'
dir_img_rotate = '/Users/kalinin/Desktop/Dataset/Dataset_rotate/'
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

		#исключение изображений, в которых больше одного символа и при этом соотношение сторон 1:1
		file_dir_path = dir_dataset[:-1] + file_dir
		if len(file_text) > 2:
			with Image.open(file_dir_path) as im:
				width, height = im.size
			ratio = width/height			
			ratio_mean += ratio
			#count += 1

			if ratio < 1:
				count_rotate += 1
				new_dir = dir_img_rotate + file_name

				if os.path.exists(new_dir):
					print('файл уже существует! запись под новым именем', file_name)
					file_name = '(' + str(count_rotate) + ')' + file_name
					new_dir = dir_img_rotate + file_name
				copyfile(file_dir_path, new_dir)

print(ratio_mean/count)
print(count)
print(count_rotate)
