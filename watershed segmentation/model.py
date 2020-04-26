import numpy as np
from skimage import color
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
        self.sigma = 2
        self.dist_threshold = 1
        self.img_path = 'shadedobj.png'
        self.img = cv.imread(self.img_path, cv.IMREAD_REDUCED_GRAYSCALE_2)

    def extract_features(self):
        edges = canny(self.img, sigma=self.sigma)
        plt.imshow(edges)
        plt.savefig('edges.png')
        plt.show()

        self.edt = distance_transform_edt(~edges)
        self.localMax = peak_local_max(
                        self.edt,
                        indices=False,
                        min_distance=self.dist_threshold
                        )
        self.markers = label(self.localMax)

    def segmentation(self):
        self.extract_features()

        labels = watershed(-self.edt, self.markers)
        plt.imshow(mark_boundaries(self.img, labels))
        plt.savefig('boundaries.png')
        plt.show()

        regions = regionprops(labels, intensity_image=self.img)
        region_means = [r.mean_intensity for r in regions]
        plt.hist(region_means, bins=20)
        plt.plot()

        kmeans = KMeans(n_clusters=2)
        region_means = np.array(region_means).reshape(-1, 1)
        kmeans.fit(region_means)

        pred_regions = kmeans.predict(region_means)

        classified_labels = labels.copy()
        for pred_region, region in zip(pred_regions, regions):
            classified_labels[tuple(region.coords.T)] = pred_region

        idx0 = (classified_labels == 0)
        idx1 = (classified_labels != 0)
        classified_labels[idx0] = 255
        classified_labels[idx1] = 0
        plt.imshow(classified_labels, cmap='gray')
        plt.savefig('segmented_image.png')
        plt.show()


if __name__ == "__main__":
    model = WatershedSegmentation()
    model.segmentation()