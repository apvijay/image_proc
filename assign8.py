#!/usr/bin/env python
""" 
  Assignment : 8
  Created by : Vijay
  Date       : Mar 12, 2012
  Topics     : Salt and pepper noise, Median filter,
               Histogram modification
"""
import Img
from Img import plot_fig, plot_hist, stem_hist
import matplotlib.pyplot as plt
import numpy as np
  
def f1():
  import pgm
  #imgarray,w,h = pgm.pgmread('histo_inp_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_inp_image2.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image2.pgm')
  imgarray,w,h = pgm.pgmread('lena.pgm')
  #imgarray,w,h = pgm.pgmread('fourier_transform.pgm')
  imgs = Img.Img(imgarray)
  imgt = imgs.add_salt_pepper(0.65)
  imgt1 = imgt.median_filter(np.ones((3,3)))
  imgt1 = imgt1.median_filter(np.ones((5,5)))
  #imgt1 = imgt.median_filter(np.ones((7,7)))
  #imgt1 = imgt.median_filter(np.array([[2,2,2],[2,1,2],[2,2,2]]))
  k = Img.create_gauss_kernel(Img.find_gauss_support(0.75), 0.75)
  imgt2 = imgt.conv(k)
  plot_fig(imgs.pix, 'Original image')
  plot_fig(imgt.pix, 'Added salt and pepper noise')
  plot_fig(imgt1.pix, 'Median filtered image')
  plot_fig(imgt2.pix, 'Gaussian LP filtered image')
  #plot_hist(imgs, 'Original image')
  #plot_hist(imgt, 'Added salt and pepper noise')
  #plot_hist(imgt1, 'Median filtered image')
  #plot_hist(imgt2, 'Gaussian LP filtered image')
  print np.sum((imgs.pix - imgt1.pix)**2)

def f2():
  import pgm
  #
  # imgs is the image to be modified
  # imgt is the reference image
  #
  imgarray,w,h = pgm.pgmread('histo_inp_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_inp_image2.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image2.pgm')
  #imgarray,w,h = pgm.pgmread('lena.pgm')
  #imgarray,w,h = pgm.pgmread('fourier_transform.pgm')
  #imgarray,w,h = pgm.pgmread('Fourier.pgm')
  imgs = Img.Img(imgarray)
  #
  #imgarray,w,h = pgm.pgmread('histo_inp_image.pgm')
  imgarray,w,h = pgm.pgmread('histo_ref_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_inp_image2.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image2.pgm')
  #imgarray,w,h = pgm.pgmread('lena.pgm')
  #imgarray,w,h = pgm.pgmread('fourier_transform.pgm')
  #imgarray,w,h = pgm.pgmread('Fourier.pgm')
  imgt = Img.Img(imgarray)
  #
  imgt1 = imgs.modify_hist(imgt)
  plot_fig(imgs.pix, 'Original image')
  plot_fig(imgt.pix, 'Reference image')
  plot_fig(imgt1.pix, 'Modified original image')
  stem_hist(imgs, 'Original image')
  stem_hist(imgt, 'Reference image')
  stem_hist(imgt1, 'Modified original image')
  print np.sum((imgs.pix - imgt1.pix)**2)

def f3():
  import pgm
  imgarray,w,h = pgm.pgmread('histo_inp_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image.pgm')
  #imgarray,w,h = pgm.pgmread('histo_inp_image2.pgm')
  #imgarray,w,h = pgm.pgmread('histo_ref_image2.pgm')
  #imgarray,w,h = pgm.pgmread('lena.pgm')
  #imgarray,w,h = pgm.pgmread('fourier_transform.pgm')
  #imgarray,w,h = pgm.pgmread('Fourier.pgm')
  #imgarray,w,h = pgm.pgmread('academy.pgm')
  imgs = Img.Img(imgarray)
  imgt = imgs.hist_eq()
  plot_fig(imgs.pix, 'Original image')
  plot_fig(imgt.pix, 'Histogram equalized image')
  stem_hist(imgs, 'Original image')
  stem_hist(imgt, 'Histogram equalized image')
  print np.sum((imgs.pix - imgt.pix)**2)

def g1():
  import pgm
  imgarray,w,h = pgm.pgmread('test.pgm')
  imgs = Img.Img(imgarray)
  imgt1 = imgs.median_filter(np.ones((3,3)))
  pgm.pgmwrite(imgt1.pix, 'test2.pgm')

def g2():
  x = np.array([[1,2],[3,4]])
  r = np.array([[4,3],[2,1]])
  x1 = x.reshape(-1)
  r1 = r.reshape(-1)
  #out = np.zeros(np.sum(r1))
  out = np.array([])
  for ind, elem in enumerate(x1):
    out = np.append(out, [elem] * r1[ind])
  print out, type(out)

if __name__ == '__main__':
  #f1()
  #f2()
  f3()
  plt.show()
