import numpy as np
import argparse
import os
from os import path, makedirs
from tqdm import tqdm 
import random
import shutil

def main(image_path, size, suffix):
	print("Loading files ...")
	image_list = np.loadtxt(image_path, delimiter=' ', dtype=np.str)
	print("Finished loading ...")

	dic = {}
	for line in tqdm(image_list):
		subject = line.split('/')[-1].split('_')[0]
		if subject not in dic:
			dic[subject] = []
		dic[subject].append(line)

	result = []
	for value in tqdm(dic.values()):
		result.extend(random.sample(value, size))

	save_path = path.join('/afs/crc.nd.edu/user/k/kzhang4/face_clustering', f'mask_list_{suffix}.txt')
	
	np.savetxt(save_path, result[:size*50], delimiter=' ', fmt='%s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creat list for face clustering')
    parser.add_argument('--image_path', '-i', help='File with image list.', default='/afs/crc.nd.edu/user/k/kzhang4/Shared/MORPH_Segmentation/MORPH_Segmentation/C_M.txt')
    parser.add_argument('--size', '-s', type=int, help='Size of sample from each subject.', default=1)
    parser.add_argument('--suffix', '-x', help='Suffix of the output file.', default=1)

    args = parser.parse_args()

    main(args.image_path, args.size, args.suffix)



