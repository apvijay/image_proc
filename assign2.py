#!/usr/bin/env python
""" 
  Assignment : 2
  Created by : Vijay
  Date       : Jan 16, 2011
  Topics     : Convolution with Gaussian kernel, Scaling of image
"""
import Img
import pgm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import shape 

def f1():
  """ Convolves image and kernel """
  (imgarr,w,h) = pgm.pgmread('cameraman.pgm')
  imgs = Img.Img(imgarr)
  k1 = Img.create_gauss_kernel(Img.find_gauss_support(0.8))
  k2 = Img.create_gauss_kernel(Img.find_gauss_support(1.2))
  k3 = Img.create_gauss_kernel(Img.find_gauss_support(1.6))
  imgt1 = imgs.conv(k1)
  imgt2 = imgs.conv(k2)
  imgt3 = imgs.conv(k3)
#  print imgt1.col, imgt1.row
#  print imgt2.col, imgt2.row
#  print imgt3.col, imgt3.row
  plt.figure(1)
  plt.imshow(imgs.pix, cmap=cm.gray)
  plt.title('Original image')
  plt.figure(2)
  plt.imshow(imgt1.pix, cmap=cm.gray)
  plt.title('Img convolved with Gaussian kernel' + str(shape(k1)))
  plt.figure(3)
  plt.imshow(imgt2.pix, cmap=cm.gray)
  plt.title('Img convolved with Gaussian kernel' + str(shape(k2)))
  plt.figure(4)
  plt.imshow(imgt3.pix, cmap=cm.gray)
  plt.title('Img convolved with Gaussian kernel' + str(shape(k3)))
  plt.show()

def f2():
  """ Scale image """
  factor = (0.8, 0.8)
  (imgarr,w,h) = pgm.pgmread('cameraman.pgm')
  imgs = Img.Img(imgarr)
  imgt1 = imgs.scale(factor)
  imgt2 = imgs.scale(factor, type='bilinear')
  plt.figure(1)
  plt.imshow(imgs.pix, cmap=cm.gray)
  plt.title('Original image')
  plt.figure(2)
  plt.imshow(imgt1.pix, cmap=cm.gray)
  plt.title('Scale factor = ' + str(factor))
  plt.figure(3)
  plt.imshow(imgt2.pix, cmap=cm.gray)
  plt.title('Scale factor = ' + str(factor) + 
            '\nwith bilinear interpolation')
  plt.show()

if __name__ == '__main__':
#  f1()
  f2()
