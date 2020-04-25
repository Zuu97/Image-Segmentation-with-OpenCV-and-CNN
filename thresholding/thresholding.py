import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class ImageThresholding(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.upper_bound = 255
        self.lower_bound = 0
        self.img_path = 'shadedobj.png'
        self.color_img = cv.imread(self.img_path, cv.IMREAD_REDUCED_COLOR_2)
        self.gray_img = cv.cvtColor(self.color_img,cv.COLOR_BGR2GRAY)

        if not os.path.exists(str(self.threshold)):
            os.mkdir(str(self.threshold))

    def numpy_threshold(self):
        upper = 255
        lower = 0
        thresh = np.where(
                    self.gray_img > self.threshold,
                    self.upper_bound,
                    self.lower_bound
                        ).astype(np.uint8)
        return thresh

    def opencv_threshold(self):
        ret,thresh = cv.threshold(
                                self.gray_img,
                                self.threshold,
                                self.upper_bound,
                                cv.THRESH_BINARY
                                )
        return thresh

    def plot_hist(self):
        hist = cv.calcHist([self.gray_img], [0], None, [self.upper_bound+1], [self.lower_bound,self.upper_bound+1])
        plt.plot(hist)
        plt.xlim([self.lower_bound,self.upper_bound+1])
        plt.title('Histogram')
        plt.savefig('histogram.png')
        plt.show()

    def segmentation(self):
        self.plot_hist()
        segment1 = self.numpy_threshold()
        segment2 = self.opencv_threshold()

        plt.title('Original Image')
        plt.imshow(self.color_img)
        plt.show()

        plt.title('Gray Image')
        plt.imshow(self.gray_img, cmap='gray')
        plt.savefig('gray_image.png')
        plt.show()

        plt.title('Thresholding Using Numpy')
        plt.imshow(segment1, cmap='gray')
        plt.savefig(str(self.threshold)+'/numpy_threshold.png')
        plt.show()

        plt.title('Thresholding Using OpenCV')
        plt.imshow(segment2, cmap='gray')
        plt.savefig(str(self.threshold)+'/opencv_threshold.png')
        plt.show()


if __name__ == "__main__":
    segmenter = ImageThresholding(50)
    segmenter.segmentation()