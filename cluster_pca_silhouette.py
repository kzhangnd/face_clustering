import numpy as np
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import path, makedirs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from multiprocessing import Pool
import os


class Eigen():
    def __init__(self):
        self.lamb = None
        self.u = None
        self.variance_proportion = None
        self.mean_vector = None


class EigenFaces():
    def __init__(self, image_path, mask):
        image_list = np.sort(np.loadtxt(image_path, dtype=np.str))
        self.total_images = len(image_list)
        self.image_size = 112

        # TEST WITH ARCFACE FEATURES INSTEAD (UNCOMMNENT AND COMMENT BOTTOM TO SWITCH)
        # self.images = np.empty(shape=(self.total_images, 512), dtype='float64')
        # for idx, image_path in tqdm(enumerate(image_list)):
        #     self.images[idx] = np.load(image_path)

        self.images = np.empty(shape=(self.total_images, self.image_size * self.image_size), dtype='float64')

        if mask is not None:
            mask_img = None
            if path.isfile(mask):
                if mask is not None:
                    mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        for idx, image_path in tqdm(enumerate(image_list)):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if mask is not None:
                if path.isfile(mask):
                    image[mask_img == 0] = 0
                else:
                    parts = image_path.split('/')
                    mask_path = path.join(mask, parts[-3], parts[-2], parts[-1][:-4] + '_mask.pdf')
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask_img[np.logical_and(mask_img > 1, mask_img <= 13)] = 1
                    mask_img[mask_img != 1] = 0

                    mask_img = cv2.resize(mask_img, (224, 224))

                    image[mask_img == 0] = 0

            image = cv2.resize(image, (self.image_size, self.image_size)).astype(float)

            image = image.flatten()
            self.images[idx, :] = image

    def get_eigen(self, energy):
        self.pca = PCA(energy)
        self.low_dim_data = self.pca.fit_transform(self.images)
        # self.low_dim_data = make_blobs(169, 80)[0]
        self.K = self.pca.n_components_
        print(f"Number of Components to get {energy} variance: {self.K}")

    def silhouette(self, n_clusters):
        cluster = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = cluster.fit_predict(self.low_dim_data)

        davi_avg = davies_bouldin_score(self.low_dim_data, cluster_labels)
        silh_avg = silhouette_score(self.low_dim_data, cluster_labels, metric='cosine')

        return n_clusters, silh_avg, davi_avg

    def get_cluster_number_mp(self, max_clusters, start, end, increment=1):
        cluster_silh = np.ones(shape=max_clusters) * float('-inf')
        cluster_davi = np.ones(shape=max_clusters) * float('inf')
        cluster_range = range(start, end, increment)

        pool = Pool(os.cpu_count())

        for n_clusters, silh_avg, davi_avg in tqdm(pool.imap_unordered(self.silhouette, cluster_range)):
            cluster_silh[n_clusters] = silh_avg
            cluster_davi[n_clusters] = davi_avg

        scores = np.argsort(cluster_davi)
        best = scores[0]
        second_best = scores[1]

        print(f'Best cluster at {best} with score {cluster_davi[best]:.4f}')
        # print(f'Second best cluster at {second_best} with score {cluster_davi[second_best]:.4f}')

        scores = np.argsort(cluster_silh)
        best = scores[-1]
        # second_best = scores[-2]

        print(f'Best cluster at {best} with score {cluster_silh[best]:.4f}')
        # print(f'Second best cluster at {second_best} with score {cluster_silh[second_best]:.4f}')

        # if increment != 1:
        #     start = min(best, second_best)
        #     end = min(max(best, second_best) + 1, max_clusters)

        #     self.get_cluster_number_mp(max_clusters, start, end, int(increment / 10))

    def reconstruct_images(self, label, dest, number=None):
        approximation = self.pca.inverse_transform(
            self.low_dim_data).reshape(-1, self.image_size, self.image_size)

        if number is None:
            number = len(approximation)

        num_row_x = num_row_y = int(np.floor(np.sqrt(number - 1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(
                np.hstack([self.images[ii].reshape(112, 112), approximation[ii]]), cmap=plt.cm.gray)
            # cv2.imwrite(path.join(dest, f'image_{ii}_original.jpg'), self.images[ii].reshape(112, 112))
            # cv2.imwrite(path.join(dest, f'image_{ii}_reconstructed.jpg'), approximation[ii])
            # axarr[div, rem].set_title("%.6f" % self.pca.explained_variance_ratio_[ii])
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x * num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

        plt.tight_layout(pad=0.2)
        plt.savefig(path.join(dest, f'{label}_reconstruction.pdf'))
        plt.close()

    def plot_eigen_vectors(self, label, dest, number=-1):
        if number < 0:
            number = self.pca.n_components_

        eigenfaces = self.pca.components_.reshape((self.pca.n_components_,
                                                   self.image_size, self.image_size))
        num_row_x = num_row_y = int(np.floor(np.sqrt(number - 1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(eigenfaces[ii], cmap=plt.cm.gray)
            axarr[div, rem].set_title("%.6f" % self.pca.explained_variance_ratio_[ii])
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x * num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

        plt.tight_layout(pad=0.2)
        plt.savefig(path.join(dest, f'{label}_eigen_faces.pdf'), dpi=150)
        plt.close()


def plot_components_energy(eigen_faces1, label1, eigen_faces2, label2, dest):
    plt.rcParams["figure.figsize"] = [6, 4.5]
    plt.rcParams['font.size'] = 12

    plt.grid(True, zorder=0, linestyle='dashed')

    max_K = max(eigen_faces1.K, eigen_faces2.K)

    eigen_faces1.pca_plot = PCA()
    eigen_faces1.pca_plot.fit(eigen_faces1.images)

    eigen_value_dist1 = np.cumsum(eigen_faces1.pca_plot.explained_variance_ratio_) / \
        np.sum(eigen_faces1.pca_plot.explained_variance_ratio_)
    eigen_value_dist1 = eigen_value_dist1[:max_K]
    plt.plot(range(1, len(eigen_value_dist1) + 1), eigen_value_dist1, label=label1, color='r')

    eigen_faces2.pca_plot = PCA()
    eigen_faces2.pca_plot.fit(eigen_faces2.images)

    eigen_value_dist2 = np.cumsum(eigen_faces2.pca_plot.explained_variance_ratio_) / \
        np.sum(eigen_faces2.pca_plot.explained_variance_ratio_)
    eigen_value_dist2 = eigen_value_dist2[:max_K]

    plt.plot(range(1, len(eigen_value_dist2) + 1), eigen_value_dist2, label=label2, color='b')

    start1 = np.where(eigen_value_dist1 >= 0.5)[0][0]
    start2 = np.where(eigen_value_dist2 >= 0.5)[0][0]
    min_x = min(start1, start2)
    max_x = max(len(eigen_value_dist1), len(eigen_value_dist2)) + 1
    # min_y = min(eigen_value_dist1[0], eigen_value_dist2[0])

    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([min_x, max_x])
    plt.ylim([0.5, 0.99])
    plt.xlabel('Number of Components')
    plt.ylabel('Percentage of Variance')
    plt.tight_layout(pad=0.2)
    plt.savefig(path.join(dest, f'{label1}_{label2}_pcs.pdf'), dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create average mask based on skin prediction.')
    parser.add_argument('--image_path1', '-i1', help='File with image list.')
    parser.add_argument('-label1', '-l1', help='Label 1.')
    parser.add_argument('--image_path2', '-i2', help='File with image list.')
    parser.add_argument('-label2', '-l2', help='Label 2.')
    parser.add_argument('--energy', '-e', help='Percent to filter.', default=0.95)
    parser.add_argument('--dest', '-d', help='Destination.')
    parser.add_argument('--mask', '-m', help='Mask.')

    args = parser.parse_args()

    if not path.exists(args.dest):
        makedirs(args.dest)

    eigen_faces1 = EigenFaces(args.image_path1, args.mask)
    eigen_faces1.get_eigen(float(args.energy))
    eigen_faces1.get_cluster_number_mp(len(eigen_faces1.images), 2,
                                       len(eigen_faces1.images), 1)

    if args.image_path2 is not None:
        eigen_faces2 = EigenFaces(args.image_path2, args.mask)
        eigen_faces2.get_eigen(float(args.energy))
        eigen_faces2.get_cluster_number_mp(len(eigen_faces2.images), 2,
                                           len(eigen_faces2.images), 1)
