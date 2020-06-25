import json
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from os import path, makedirs
from collections import Counter
from tqdm import tqdm 
import random
import shutil

#male = set(np.loadtxt('stage_1/race_gender_above_75_more_5/Caucasian_Male.txt', delimiter=' ', dtype=np.str))
#female = set(np.loadtxt('stage_1/race_gender_above_75_more_5/Caucasian_Female.txt', delimiter=' ', dtype=np.str))
print("Loading files ...")
mask_list = np.loadtxt('mask_list_1.txt', delimiter=' ', dtype=np.str)
#deletion = set(list(np.loadtxt('stage_4/deletion.txt', delimiter=' ', dtype=np.str)))
#label = np.loadtxt('stage_2/match/Overall_labels.txt', delimiter=' ', dtype=np.str)
#image_list = np.loadtxt('stage_3/new_list.txt', delimiter=' ', dtype=np.str)
#intersection = set(list(np.loadtxt('stage_2/intersection_subject_final.txt', delimiter=' ', dtype=np.str)))
print("Finished loading ...")

head = '/afs/crc.nd.edu/user/k/kzhang4/Shared/MORPH3_Images/RetinaFace'
result = []
for line in tqdm(mask_list):
	meta = line.split('/')[-3]
	folder = line.split('/')[-2]
	name = line.split('/')[-1].split('_')[0] + '_' + line.split('/')[-1].split('_')[1] + '.JPG'
	new = path.join(head, meta, folder, name)
	if not path.isfile(new):
		print('Alert')
	result.append(new)

np.savetxt('image_list.txt', result, delimiter=' ', fmt='%s')






'''
image_list = []
with open('stage_3/feature_list.txt') as f:
	for line in f:
		image_list.append(line.strip())

deletion = set()
with open('stage_4/deletion.txt') as f:
	for line in f:
		deletion.add(line.strip())

result = []


for line in tqdm(image_list):
	label = path.join(line.split('/')[-2], line.split('/')[-1][:-4])
	if label not in deletion:
		result.append(line)
	else:
		deletion.remove(label)

print(deletion)

#np.savetxt('stage_4/post_deletion.txt', result, delimiter=' ', fmt='%s')
'''

'''

for image in tqdm(image_list):
	subject = image.split('/')[-2]
	
	image_path = path.join(subject, image.split('/')[-1])
	new_path = path.join(dest, image_path)
	destination = path.split(new_path)[0]

	if not path.exists(destination):
		makedirs(destination)

	shutil.copy(image, new_path)









#np.savetxt('stage_2/intersection.txt', np.array(list(b)), delimiter=' ', fmt='%s')
#np.savetxt('stage_1/Caucasian_subset/female.txt', np.array(f), delimiter=' ', fmt='%s')




#plt.xlim(0.4, 1.1)

plt.bar(langs, data)

#plt.xticks(np.arange(0.4, 1.1, 0.05))
plt.title('Race Classification')
#plt.show()
plt.savefig('race_classification.png', dpi=500)

'''
#np.savetxt('subject_info.txt', array, delimiter=' ', fmt='%s')

