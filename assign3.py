#!/usr/bin/env python
""" 
  Assignment : 3
  Created by : Vijay
  Date       : Jan 23, 2011
  Topics     : (i)  Space variant blur for Gauss varying sigma
               (ii) Compare constant sigma using space variant blur and
                    convolution (uncomment four lines)
"""
import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

show_plot = True

def f1(x, y, N):
  a = x
  b = -N**2/(2*np.log(y/a))
  k = Image.create_sigma(a, b, (N,N))
  if show_plot == True:
    print k[0][0], k[0][N-1], k[N-1][0], k[N-1][N-1], k[N/2][N/2]
    x = np.arange(N)
    y = np.arange(N)
    X,Y = np.meshgrid(x,y)
    fig1 = plt.figure()
    plt.title('Variation of $\sigma$ as Gaussian')
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, k, cmap=cm.jet)
    fig2 = plt.figure()
    plt.title('Variation of $\sigma$ as Gaussian (2D)')
    plt.imshow(k)
    plt.colorbar()
  return k

def f2(sigma):
  import pgm
  (imgarray,w,h) = pgm.pgmread('cameraman.pgm')
  imgs = Image.Image(imgarray)
  suppMat, imgt = imgs.lsv_blur(sigma)
  if show_plot == True:
    plt.figure()
    plt.imshow(suppMat)
    plt.title('Kernel size variation')
    plt.colorbar()
  plt.figure()
  plt.imshow(imgs.pix, cmap=cm.gray)
  plt.title('Original image')
  plt.figure()
  plt.imshow(imgt.pix, cmap=cm.gray)
  plt.title('Space variant blur')
  pgm.pgmwrite(imgt.pix, 'a3_lsv_blur.pgm')
  #
  imgt1 = f3(imgs)
  print 'Kernel size: ', suppMat[0][0]
  print 'Error : ', np.sum((imgt1.pix - imgt.pix)**2) / 256

def f3(imgs):
  """ Convolves image and kernel """
  k = Image.create_gauss_kernel(Image.find_gauss_support(0.8), 0.8)
  imgt = imgs.conv(k)
  plt.figure()
  plt.imshow(imgt.pix, cmap=cm.gray)
  plt.title('Image convolved with Gaussian kernel' + str(np.shape(k)))
  return imgt

if __name__ == '__main__':
  from datetime import datetime
  print datetime.time(datetime.now())
  N = 256
#  sigma = f1(2, 0.05, N)
  sigma = np.ones((N,N)) * 0.8
  f2(sigma)
  print datetime.time(datetime.now())  
  plt.show()
