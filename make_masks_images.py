import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
from os import path, makedirs
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def average_image(source, dest, percent, mask):
    source_file = np.sort(np.loadtxt(source, dtype=np.str))

    total_images = len(source_file)
    average_image = None

    if mask is not None:
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    for image_path in tqdm(source_file):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # face masks
        # image[image == 1] = 0
        image[np.logical_and(image > 1, image <= 13)] = 1
        image[image > 13] = 0
        # image[image != 17] = 0
        # image[image == 17] = 1

        image = cv2.resize(image, (224, 224))
        if mask is not None:
            image[mask == 0] = 0
            image = cv2.resize(image, (112, 112))

        # cv2.imshow('show', image)
        # cv2.waitKey(100)

        image = np.float32(image)

        if average_image is None:
            average_image = image
        else:
            average_image += image

    average_image /= total_images

    heatmap = plt.imshow(average_image, cmap='hot', vmin=0., vmax=1.)
    plt.colorbar(heatmap)
    plt.tight_layout(pad=0.2)
    plt.savefig(path.join(dest, path.split(source)[1][:-4] + '_average_heatmap.png'), dpi=150)

    average_image[average_image >= percent] = 255
    average_image[average_image != 255] = 0

    cv2.imwrite(path.join(dest, path.split(source)[1][:-4] + '_average_image.png'), average_image)
    np.save(path.join(dest, path.split(source)[1][:-4] + '_average_array.npy'), average_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create average mask based on skin prediction.')
    parser.add_argument('--source', '-s', help='File with image list.')
    parser.add_argument('--dest', '-d', help='Folder to save average face image.')
    parser.add_argument('--percent', '-p', help='Percent to filter.')
    parser.add_argument('--mask', '-m', help='Mask.')

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    average_image(args.source, args.dest, float(args.percent), args.mask)
