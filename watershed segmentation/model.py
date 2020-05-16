import numpy as np
from skimage import color
from skimage.filters import median
from skimage.morphology import watershed
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.feature import canny, peak_local_max
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2 as cv

class WatershedSegmentation(object):
    def __init__(self):
        self.sigma = 2.5
        self.dist_threshold = 2
        self.img_path = 'defective_weld.tif'
        self.color_img = cv.imread(self.img_path, cv.IMREAD_REDUCED_COLOR_2)
        self.img = cv.imread(self.img_path, cv.IMREAD_REDUCED_GRAYSCALE_2)/255.0
        # self.img = median(self.img, selem=np.ones((1, 1)))

    def extract_features(self):
        edges = canny(self.img, sigma=self.sigma)

        self.edt = distance_transform_edt(~edges)
        self.localMax = peak_local_max(
                        self.edt,
                        indices=False,
                        min_distance=self.dist_threshold
                        )
        self.markers = label(self.localMax)
        self.edges = edges

    def segmentation(self):
        self.extract_features()
        labels = watershed(-self.edt, self.markers)
        regions = regionprops(labels, intensity_image=self.img)
        region_means = [r.mean_intensity for r in regions]

        kmeans = KMeans(n_clusters=2)
        region_means = np.array(region_means).reshape(-1, 1)
        kmeans.fit(region_means)
        pred_regions = kmeans.predict(region_means)

        classified_labels = labels.copy()
        for pred_region, region in zip(pred_regions, regions):
            classified_labels[tuple(region.coords.T)] = pred_region

        idx0 = (classified_labels == 0)
        idx1 = (classified_labels != 0)
        classified_labels[idx0] = 1
        classified_labels[idx1] = 0

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(221)
        ax1.imshow(self.color_img.astype(np.uint8))
        ax1.title.set_text('Originl Image')
        ax2 = fig.add_subplot(222)
        ax2.imshow(self.edges)
        ax2.title.set_text('edges')
        ax3 = fig.add_subplot(223)
        ax3.imshow(mark_boundaries(self.img, labels))
        ax3.title.set_text('boundaries')
        ax4 = fig.add_subplot(224)
        ax4.imshow(classified_labels, cmap='gray')
        ax4.title.set_text('Output Image')
        plt.savefig('results_'+self.img_path)
        plt.show()


if __name__ == "__main__":
    model = WatershedSegmentation()
    model.segmentation()