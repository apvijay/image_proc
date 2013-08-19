#!/usr/bin/env python
""" 
  Assignment : 7
  Created by : Vijay
  Date       : Feb 27, 2011
  Topics     : Bilateral filter
"""
# Observations:
# 1. sigmaR is large(100 or 300) wrt overall range of values in image
#    (0-255): Range component has little effect for small sigmaD
# 2. sigmaR is small(1 or 10): Range filter dominates perceptually,
#    because it preserves the edges (no blurring due to domain filter)
# 3. Comparing (3,100) and (5,100): Latter has a very broad Gaussian, 
#    therefore essentially a range filter. Latter is hazy but yet 
#    crispier than the former
import Img
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from string import capitalize as caps

def f1(imgs, sigmaD, sigmaR, filtType='bilateral'):
  import pgm
  imgt = imgs.bilateral_filter(sigmaD, sigmaR, filtType)
  plotTitle = caps(filtType) + ' filter with sigmaD=' + str(sigmaD) +\
              ' and sigmaR=' + str(sigmaR)
  plot_fig(imgt.pix, plotTitle)
#  plot_hist(imgt, plotTitle)
#  print np.sum((imgt.pix - imgs.pix)**2)
  filename = 'academy_' + filtType + '_' + str(sigmaD) + '_' +\
              str(sigmaR) + '.pgm'
  pgm.pgmwrite(np.int32(imgt.pix), filename)
  print '.'

def add_noise(imgarr):
  r, c = np.shape(imgarr)
  imgarr = np.float32(imgarr) + 0.05*255*np.random.randn(r,c)
  imgarr[np.where(imgarr > 255)] = 255
  imgarr[np.where(imgarr < 0)] = 0
  imgarr = np.int32(imgarr)
  return imgarr

def plot_hist(img, plotTitle):
  imgarray = img.calc_hist()
  plt.figure()
  plt.plot(imgarray)
  plt.title(plotTitle)

def plot_fig(imgarr, plotTitle):
  plt.figure()
  plt.imshow(imgarr, cmap=cm.gray)
  plt.title(plotTitle)

if __name__ == '__main__':
  import pgm
  (imgarray,w,h) = pgm.pgmread('academy.pgm')
  img = Img.Img(imgarray)
#  plot_fig(np.int32(img.pix), 'Original mage')
#  plot_hist(img, 'Original image')
  imgarray = add_noise(imgarray)
  imgs = Img.Img(imgarray)
  plot_fig(imgs.pix, 'Image with additive Gaussian with sigma=5')
  pgm.pgmwrite(np.int32(imgs.pix), 'academy_with_noise.pgm')
#  plot_hist(img, 'Noise image')
  ##
#  f1(imgs, 1, 1)
#  f1(imgs, 1, 10)
#  f1(imgs, 1, 100)
#  f1(imgs, 1, 300)
  ##
#  f1(imgs, 3, 1)
#  f1(imgs, 3, 10)
  f1(imgs, 3, 50)
#  f1(imgs, 3, 100)
#  f1(imgs, 3, 300)
  ##
#  f1(imgs, 5, 1)
#  f1(imgs, 5, 10)
  f1(imgs, 5, 50)
#  f1(imgs, 5, 100)
#  f1(imgs, 5, 300)
  ##
#  f1(imgs, 10, 1)
#  f1(imgs, 10, 10)
  f1(imgs, 10, 50)
#  f1(imgs, 10, 100)
#  f1(imgs, 10, 300)
  ##
#  f1(imgs, 3, 1, filtType = 'domain')
#  f1(imgs, 5, 1, filtType = 'domain')
#  f1(imgs, 10, 1, filtType = 'domain')
#  f1(imgs, 3, 100, filtType = 'range')
#  f1(imgs, 5, 100, filtType = 'range')
#  f1(imgs, 10, 100, filtType = 'range')
  plt.show()
