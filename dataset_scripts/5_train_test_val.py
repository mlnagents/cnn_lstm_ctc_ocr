import os
import sys
import operator
import random

dir_dataset = '/Users/kalinin/Desktop/Dataset'
dir_txt = ['BornDigitalImages_new.txt', 'COCO_new.txt', 'FocusedSceneText_new.txt', 'Sport_new.txt',
		   'IIIT5K_new.txt', 'IncidentalSceneText_new.txt', 'MJSynth_new.txt', 'MLT_new.txt', 'Medicine_new.txt']

dir_train_txt = 'dir_train.txt'
dir_test_txt = 'dir_test.txt'
dir_val_txt = 'dir_val.txt'

count = 0
count_train = 0
count_test = 0
count_val = 0

train_share = 0.8
test_share = 0.1
val_share = 1 - (train_share + test_share)

train_txt = open(os.path.join(dir_dataset, dir_train_txt), 'w')
test_txt = open(os.path.join(dir_dataset, dir_test_txt), 'w')
val_txt = open(os.path.join(dir_dataset, dir_val_txt), 'w')

for dataset in dir_txt:
	dir_dataset_txt = os.path.join(dir_dataset, dataset)
	print(dataset, '...')
	with open(dir_dataset_txt, 'r') as t:
		content = t.readlines()

	for line in content:
		count += 1
		line = line.strip()
		chance = random.uniform(0, 1)
		if chance < train_share:
			count_train += 1
			train_txt.write(line + '\n')
		elif chance < train_share + test_share:
			count_test += 1
			test_txt.write(line + '\n')
		else:
			count_val += 1
			val_txt.write(line + '\n')

train_txt.close()
test_txt.close()
val_txt.close()

print('Done')
print('всего изображений:', count)
print('train:', count_train)
print('test', count_test)
print('val', count_val)