import math
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import cv2 as cv

class BasicLevelSet(object):
    def __init__(self):
        self.num_iterations = 200
        self.color_img = cv.cvtColor(
                            cv.resize(
                                    cv.imread('Lenna.png'),
                                    (240, 240)
                                    ),
                            cv.COLOR_BGR2RGB)

        self.gray_img = cv.resize(
                            cv.imread('two_obj.png', cv.IMREAD_GRAYSCALE),
                            (240, 240)
                            )
        self.img = self.gray_img - np.mean(self.gray_img)
        self.img_smooth = cv.GaussianBlur(self.img ,(3, 3), 0)

    @staticmethod
    def grad(x):
        return np.array(np.gradient(x))

    @staticmethod
    def norm(x, axis=0):
        return np.sqrt(np.sum(np.square(x), axis=axis))

    def stopping_fun(self):
        return 1. / (1. + BasicLevelSet.norm(BasicLevelSet.grad(self.img_smooth))**2)

    def initialize_phi(self, initshape ='rectangle'):
        phi = -1.0*np.ones(self.img_smooth.shape[:2])
        if initshape == 'rectangle':
            phi[20:-20, 20:-20] = 1.
        else: # Circle
            r = 48
            M, N = phi.shape
            a = M/2
            b = N/2
            y,x = np.ogrid[-a:M-a, -b:N-b]
            mask = x*x + y*y <= r*r
            phi[mask] = 1.
        self.phi = phi

    def plot_initials(self):
        self.initialize_phi()
        g = self.stopping_fun()
        g[g < 0.1] = 0.0

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(131)
        ax1.imshow(self.img, cmap='gray')
        ax1.title.set_text('Image')
        ax2 = fig1.add_subplot(132)
        ax2.imshow(g, cmap=cm.coolwarm)
        ax2.title.set_text('F')
        self.ax3 = fig1.add_subplot(133)
        self.ax3.imshow(self.phi, cmap=cm.coolwarm)
        self.ax3.title.set_text(r"$\phi$")
        plt.savefig('Lenna_phi.png')
        plt.pause(1.)

        # Plotting the (clipped) level-set function phi and the image with level curve
        self.fig2 = plt.figure(figsize=(15,5))
        self.ax3 = self.fig2.add_subplot(131, projection='3d')
        self.ax3.view_init(elev=30., azim=-210)
        M, N = self.phi.shape
        self.X = np.arange(0, N, 1)
        self.Y = np.arange(0, M, 1)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        self.ax4 = self.fig2.add_subplot(132)
        self.ax4.imshow(self.gray_img, cmap='gray')

        self.g = g

    def segmentation(self):
        self.plot_initials()
        dt = 1.
        ims = []
        for i in range(self.num_iterations+1):
            grad_phi = BasicLevelSet.grad(self.phi)
            grad_phi_norm = BasicLevelSet.norm(grad_phi)
            phi_t = - self.g * grad_phi_norm
            self.phi = self.phi + dt * phi_t

            self.ax3.cla()
            surf = self.ax3.plot_surface(self.X, self.Y, np.clip(self.phi, -3.0, 3.0),
                                         cmap=cm.bwr, linewidth=0, antialiased=False)
            plt.pause(0.1)
            for c in self.ax4.collections:
                c.remove()
            self.ax4.contour(self.phi, levels=[0], colors=['yellow'])
            self.fig2.suptitle("Iterations {:d}".format(i))
            if i == self.num_iterations - 1:
                ax5 = self.fig2.add_subplot(133)
                ax5.imshow(self.color_img.astype(np.uint8))
                ax5.title.set_text('Resized Color Input')
                plt.savefig('Lenna_results.png')
                plt.show()

if __name__ == "__main__":
    model = BasicLevelSet()
    model.segmentation()