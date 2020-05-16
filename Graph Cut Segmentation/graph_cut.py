import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from collections import defaultdict
from numpy import linalg as la
import numpy as np
import cv2 as cv
import maxflow

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.circle(timg,(x,y),10,(255,0,0),-1)
            else:
                cv.circle(timg,(x,y),10,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.circle(timg,(x,y),10,(255,0,0),-1)
        else:
            cv.circle(timg,(x,y),10,(0,0,255),-1)

def compute_pdfs(img,seeds,timg):
    scribbles = seeds
    imageo = np.zeros(img.shape)
    # separately store background and foreground scribble pixels in the dictionary comps
    comps = defaultdict(lambda:np.array([]).reshape(0,1))
    mu, Sigma = {}, {}
    for (i, j) in scribbles:
      clr = timg[i,j,:]
      comps[tuple(clr)] = np.vstack([comps[tuple(clr)], img[i,j]])
    # compute MLE parameters for Gaussians
    for c in comps:
        mu[c] = np.mean(comps[c])
        Sigma[c] = np.std(comps[c])
    return (mu, Sigma)

def prob(mu,sigma,img):
    pf = norm.pdf(img,loc=mu['f'],scale=sigma['f'])
    pb = norm.pdf(img,loc=mu['b'],scale=sigma['b'])
    wiB = -np.log(pf)
    wiF = -np.log(pb)
    return wiF,wiB

def edges(img):
    sigma = np.std(img)
    s=sigma
    k=1
    r,c = img.shape
    edu = np.ones((r,c), dtype= np.double)
    edd = np.ones((r,c), dtype= np.double)
    edr = np.ones((r,c), dtype= np.double)
    edl = np.ones((r,c), dtype= np.double)
    Im=img
    for i in range(r):
        for j in range(c):
            if((i-1)>0):
                wu=k*np.exp(-(abs(Im[i,j]-Im[i-1,j]))/(2*s**2))
                edu[i,j]=wu
            if((i+1)<Im.shape[0]):
                wd=k*np.exp(-(abs(Im[i,j]-Im[i+1,j]))/(2*s**2))
                edd[i,j]=wd
            if((j-1)>0):
                wl=k*np.exp(-(abs(Im[i,j]-Im[i,j-1]))/(2*s**2))
                edl[i,j]=wl
            if((j+1)<Im.shape[1]):
                wr=k*np.exp(-(abs(Im[i,j]-Im[i,j+1]))/(2*s**2))
                edr[i,j]=wr

    return edu,edd,edl,edr

def graph(img,edu,edd,edl,edr):
    g = maxflow.Graph[float](0,0)
    nodeids = g.add_grid_nodes((img.shape[0],img.shape[1]))
    structure = np.array([[0,0,0],
                          [0,0,0],
                          [0,1,0]])
    g.add_grid_edges(nodeids,structure=structure, weights=edd,symmetric=False)
    structure = np.array([[0,0,0],
                          [0,0,1],
                          [0,0,0]])
    g.add_grid_edges(nodeids,structure=structure, weights=edr,symmetric=False)
    structure = np.array([[0,0,0],
                          [1,0,0],
                          [0,0,0]])
    g.add_grid_edges(nodeids,structure=structure, weights=edl,symmetric=False)
    structure = np.array([[0,1,0],
                          [0,0,0],
                          [0,0,0]])
    g.add_grid_edges(nodeids,structure=structure, weights=edu,symmetric=False)
    g.add_grid_tedges(nodeids,wiF,wiB)
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    return img2,sgm

def plot(file,img2):
    I = cv.imread(file)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    out = np.ones((I.shape[0],I.shape[1],3))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if img2[i,j]==1:
                if len(I.shape) == 2:
                    out[i,j,0],out[i,j,1],out[i,j,2] = 1,1,1
                if len(I.shape) == 3:
                    out[i,j,0]=I[i,j,0]
                    out[i,j,1]=I[i,j,1]
                    out[i,j,2]=I[i,j,2]
            else:
                out[i,j,0],out[i,j,1],out[i,j,2] = 0,0,0

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.imshow(I,cmap='gray')
    ax1.title.set_text('Original Image')
    ax2 = fig1.add_subplot(122)
    ax2.imshow(out.astype(np.uint8))
    ax2.title.set_text('Graph Cut\nSegmented Image')
    plt.plot()


def bg_and_fg(img,timg):
    seeds = []
    fs = []
    bs = []
    r,c = img.shape
    for i in range(r):
        for j in range(c):
            if (timg[i][j] == np.array([0,0,255])).all():
                seeds.append((i,j))
                bs.append((i,j))
            elif (timg[i][j] == np.array([255,0,0])).all():
                seeds.append((i,j))
                fs.append((i,j))
    return seeds,fs,bs

def GraphCut(img,timg):
    seeds,fs,bs=bg_and_fg(img,timg)
    mu,sigma = compute_pdfs(img,seeds,timg)
    mu,sigma = {"f":mu[(255,0,0)],"b":mu[(0,0,255)]},{"f":sigma[(255,0,0)],"b":sigma[(0,0,255)]}

    wiF,wiB = prob(mu,sigma,img)
    fs = np.array(fs)
    bs = np.array(bs)
    edu,edd,edl,edr = edges(img)
    img2,sgm=graph(img,edu,edd,edl,edr)
    plot(file,img2)
    return img2


file="images/bird.jpg"
assert os.path.exists(os.path.join(os.getcwd(), file)), "Image doesn't exists"
img = cv.imread(file)  #,cv.IMREAD_GRAYSCALE
timg = cv.imread(file)
imgb=img[:,:,0]
imgg=img[:,:,1]
imgr=img[:,:,2]

cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)

while(1):
    cv.imshow('image',timg)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv.destroyAllWindows()

img=imgr

seeds,fs,bs=bg_and_fg(img,timg)
mu,sigma = compute_pdfs(img,seeds,timg)
print(mu)
print(sigma)
mu,sigma = {"f":mu[(255,0,0)],"b":mu[(0,0,255)]},{"f":sigma[(255,0,0)],"b":sigma[(0,0,255)]}
print(mu)
print(sigma)
wiF,wiB = prob(mu,sigma,img)
fs = np.array(fs)
bs = np.array(bs)
edu,edd,edl,edr = edges(img)
seg0,sgm=graph(img,edu,edd,edl,edr)
# plot(file,seg0)


img=imgg

seeds,fs,bs=bg_and_fg(img,timg)
mu,sigma = compute_pdfs(img,seeds,timg)
print(mu)
print(sigma)
mu,sigma = {"f":mu[(255,0,0)],"b":mu[(0,0,255)]},{"f":sigma[(255,0,0)],"b":sigma[(0,0,255)]}
print(mu)
print(sigma)
wiF,wiB = prob(mu,sigma,img)
fs = np.array(fs)
bs = np.array(bs)
edu,edd,edl,edr = edges(img)
seg1,sgm=graph(img,edu,edd,edl,edr)
# plot(file,seg1)


img=imgb

seeds,fs,bs=bg_and_fg(img,timg)
mu,sigma = compute_pdfs(img,seeds,timg)

print(mu)
print(sigma)
mu,sigma = {"f":mu[(255,0,0)],"b":mu[(0,0,255)]},{"f":sigma[(255,0,0)],"b":sigma[(0,0,255)]}
print(mu)
print(sigma)
wiF,wiB = prob(mu,sigma,img)
fs = np.array(fs)
bs = np.array(bs)
edu,edd,edl,edr = edges(img)
seg2,sgm=graph(img,edu,edd,edl,edr)
# plot(file,seg2)


I = cv.imread(file)
I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
out = np.ones((I.shape[0],I.shape[1],3))



seg=(seg0+seg1+seg2)/(3)
for i in range(seg0.shape[0]):
    for j in range(seg0.shape[1]):
        if (seg[i,j]>0.8):
            if len(I.shape) == 2:
                out[i,j,0],out[i,j,1],out[i,j,2] = 1,1,1
            if len(I.shape) == 3:
                out[i,j,0]=I[i,j,0]
                out[i,j,1]=I[i,j,1]
                out[i,j,2]=I[i,j,2]
        else:
            out[i,j,0],out[i,j,1],out[i,j,2] = 0,0,0


fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.imshow(I,cmap='gray')
ax1.title.set_text('Original Image')
ax2 = fig1.add_subplot(122)
ax2.imshow(out.astype(np.uint8))
ax2.title.set_text('Graph Cut\nSegmented Image')
plt.savefig("results/bird.png")
plt.show()