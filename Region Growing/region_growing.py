import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class RegionGrowing(object):
    def __init__(self, diff_threshold):
        self.diff_threshold = diff_threshold
        self.img_path = 'shadedobj.png'
        self.img = cv.imread(self.img_path, cv.IMREAD_REDUCED_GRAYSCALE_2)

    def get4neighbours(self, x, y):
        out = []
        maxx = self.img.shape[0]-1
        maxy = self.img.shape[1]-1

        #west
        outx = x
        outy = max(y-1,0)
        out.append((outx,outy))

        #north
        outx = max(x-1,0)
        outy = y
        out.append((outx,outy))

        #south
        outx = min(x+1,maxx)
        outy = y
        out.append((outx,outy))

        #east
        outx = x
        outy = min(y+1,maxy)

        out.append((outx,outy))
        return out

    def mouse_point(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print('Seed: (' + str(y) + ', ' + str(x),') ', self.img[y,x])
            self.clicks.append((y,x))

    def region_growing(self, seeds):
        checkpoints = []
        outimg = np.zeros_like(self.img)
        for seed in seeds:
            checkpoints.append([(seed[0], seed[1])])
        processed = np.zeros(self.img.shape, dtype=np.bool)

        step = 0
        while (checkpoints != [[]] * len(seeds)):
            for i in range(len(seeds)):
                if len(checkpoints[i]) > 0:
                    p = checkpoints[i].pop(0)
                    processed[p] = True
                    outimg[p[0], p[1]] = 255
                    for q in self.get4neighbours(p[0], p[1]):
                        if abs(self.img[q[0], q[1]].astype('float32')  - self.img[p[0], p[1]].astype('float32')) < self.diff_threshold:
                            outimg[q[0], q[1]] = 255
                            if not processed[q]:
                                checkpoints[i].append(q)
                            processed[q] = True

                    cv.imshow("Progress",outimg)
                    cv.waitKey(1)
                    if step % 200 == 0:
                        cv.imwrite(str(len(self.clicks))+'_seeds'+'/region_growing_'+str(step)+'.png',outimg)

            step += 1
        print("DONE")
        cv.destroyAllWindows()
        cv.imwrite(str(len(self.clicks))+'_seeds'+'/region_growing_final.png',outimg)

    def segmentation(self):
        self.clicks = []
        cv.namedWindow('Input')
        cv.setMouseCallback('Input', self.mouse_point, 0, )
        cv.imshow('Input', self.img)
        cv.waitKey(10000)
        cv.destroyAllWindows()

        if not os.path.exists(str(len(self.clicks))+'_seeds'):
            os.mkdir(str(len(self.clicks))+'_seeds')
        self.region_growing(self.clicks)


if __name__ == "__main__":
    segmenter = RegionGrowing(5)
    segmenter.segmentation()