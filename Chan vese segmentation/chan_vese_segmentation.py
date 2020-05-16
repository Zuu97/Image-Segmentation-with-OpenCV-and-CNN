import time
import sys
import os
import math
import cv2 as cv
from PIL import Image
from PIL.Image import core as _imaging
import numpy as np
from numpy import linalg
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import helpers_cv_levelset as helpers

class ChanVeseSegmentation(object):
    def __init__(self, p_image):
        self.verbose = True
        self.save_figs = True
        self.output_filetype = '.png'
        self.mu = 0.2
        self.nu = 0
        self.lambda1 = 1
        self.lambda2 = 1
        self.dt = 0.5
        self.eps = 1
        self.eta = 10e-8
        self.tol = 1e-4
        self.max_iters = 20
        self.p_image = p_image

        self.p_figs = os.path.join('figs', os.path.splitext(os.path.basename(p_image))[0])
        os.makedirs(self.p_figs, exist_ok=True)

    @staticmethod
    def noisy(img):
        image = img.copy()
        image = image/ 255.0
        row,col,ch= image.shape
        mean = 0
        var = 0.00001
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return (noisy * 255.0).astype(np.uint8)

    def load_image(self):
        color_im = cv.imread(self.p_image)
        noisy_img = ChanVeseSegmentation.noisy(color_im)
        self.im = cv.cvtColor(noisy_img, cv.COLOR_BGR2GRAY)

        fig_ = plt.figure()
        ax1 = fig_.add_subplot(131)
        ax1.set_title('Input image')
        ax1.imshow(color_im.astype(np.uint8))
        ax2 = fig_.add_subplot(132)
        ax2.set_title('Noisy image')
        ax2.imshow(noisy_img.astype(np.uint8))
        ax3 = fig_.add_subplot(133)
        ax3.set_title('Gray image')
        ax3.imshow(self.im.astype(np.uint8), cmap='gray')
        plt.savefig('figs/ostrich/add_noise.png')
        plt.show()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.im, cmap='gray')

    def A(self, lsf, i, j):
        return helpers.coeff_A(lsf, i, j, self.mu, self.eta)

    def B(self, lsf, i, j):
        return helpers.coeff_B(lsf, i, j, self.mu, self.eta)

    def H(self, t):
        return helpers.heaviside_reg(t, self.eps)

    def D(self, t):
        return helpers.delta_reg(t, self.eps)

    def segmentation(self):
        self.load_image()
        im = self.im
        M, N = im.shape
        lsf = np.zeros((M, N))
        for x1 in range(M):
            for x2 in range(N):
                lsf[x1, x2] = math.sin(math.pi/5*x1) * math.sin(math.pi/5*x2)
        I = helpers.BoundsHandler_Clamp(im)
        L = helpers.BoundsHandler_Clamp(lsf)
        self.L = L

        iter = 0
        break_loop = False
        while True:
            sys.stdout.write('iter: {:4d}'.format(iter))

            lsf_n = np.copy(lsf)
            c1, c2 = helpers.update_C_reg(I, L, self.H)

            for i in range(M):
                for j in range(N):
                    dtd = self.dt*self.D(L[i,j])
                    L[i,j] = (L[i,j] + dtd*(self.A(L,i,j)*L[i+1,j] + self.A(L,i-1,j)*L[i-1,j] +
                                            self.B(L,i,j)*L[i,j+1] + self.B(L,i,j-1)*L[i,j-1] -
                                            self.nu - self.lambda1*(I[i,j]-c1)**2 + self.lambda2*(I[i,j]-c2)**2)
                            ) / (1 + dtd*(self.A(L,i,j)+self.A(L,i-1,j)+self.B(L,i,j)+self.B(L,i,j-1)))

            if break_loop or self.save_figs:
                helpers.update_fig_contour(self.ax, lsf)
                fig_filename = 'iter_{:04d}'.format(iter)
            err = np.linalg.norm(lsf.ravel() - lsf_n.ravel(), 2) / (M*N)
            if self.verbose:
                sys.stdout.write(' || C1: {:7.2f}, C2: {:7.2f}'.format(c1, c2))
                sys.stdout.write(' | cost: {:10.4e}\n'.format(err))
            if err < self.tol:
                print('\nCONVERGED  final cost: {:f}'.format(err))
                self.ax.set_title('Convergence after {:d} iters'.format(iter))
                fig_filename += '_convergence'
                break_loop = True
            elif iter >= self.max_iters:
                print('\nHALT  exceeded max iterations')
                self.ax.set_title('Early halt after {:d} iters'.format(iter))
                fig_filename += '_earlyhalt'
                break_loop = True
            if break_loop:
                self.fig.savefig(os.path.join(self.p_figs, fig_filename + self.output_filetype))
                break

            if self.save_figs:
                self.ax.set_title('iter {:4d}'.format(iter))
                self.fig.savefig(os.path.join(self.p_figs, fig_filename + self.output_filetype))

            iter += 1

    def visualize(self):
        fig1 = plt.figure()
        ax = fig1.gca(projection='3d')

        Z = self.L.array
        # np.savetxt("Z.csv", Z, delimiter=",")
        M, N = Z.shape
        X = np.arange(0, M, 1)
        Y = np.arange(0, N, 1)
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z.T, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig1.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(self.p_figs+'/results.png')
        plt.show()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('ArgumentError: Must supply image as an argument\n'
                '  Usage:  {!s} <image-file>'.format(sys.argv[0]))

    p_image = sys.argv[1]
    model = ChanVeseSegmentation(p_image)
    model.segmentation()
    model.visualize()