#!/usr/bin/env python
""" 
  Assignment : 5
  Created by : Vijay
  Date       : Feb 13, 2011
  Topics     : 2D DFT from 1D DFT
"""
from Img import dft_2d_from_1d, idft_2d_from_1d
import numpy as np
import pgm
import matplotlib.pyplot as plt
from matplotlib import cm

def f1(filename):
  from scipy.fftpack import fftshift
  (imgarr,w,h) = pgm.pgmread(filename)
  imgarr = np.float32(imgarr)
  #
  dftArr = dft_2d_from_1d(imgarr)
  #
  idftArr = idft_2d_from_1d(dftArr)
  idftArr = np.round(idftArr.real)
  print 'Error : ', np.sum((imgarr - idftArr)**2)
  #
  dftArrAbs = np.abs(fftshift(dftArr))
  #
  plt.figure()
  plt.subplot(131)
  plt.title(filename)
  plt.imshow(imgarr, cmap=cm.gray)
  plt.colorbar()
  plt.subplot(132)
  plt.title('DFT of ' + filename)
  plt.imshow(np.int32(np.log(dftArrAbs)), cmap=cm.gray)
  plt.colorbar()
  plt.subplot(133)
  plt.title('IDFT')
  plt.imshow(np.int32(idftArr), cmap=cm.gray)
  plt.colorbar()

def f2(file1, file2):
  from scipy.fftpack import fftshift
  (imgarr1,w,h) = pgm.pgmread(file1)
  (imgarr2,w,h) = pgm.pgmread(file2)
  imgarr1 = np.float32(imgarr1)
  imgarr2 = np.float32(imgarr2)
  print imgarr1.shape, imgarr2.shape
  #
  dftArr1 = dft_2d_from_1d(imgarr1)
  dftArr2 = dft_2d_from_1d(imgarr2)
  #
  absArr1 = np.abs(dftArr1)
  absArr2 = np.abs(dftArr2)
  phaseArr1 = np.angle(dftArr1)
  phaseArr2 = np.angle(dftArr2)
  #
  newArr1 = absArr1 * np.exp(1j * phaseArr2)
  newArr2 = absArr2 * np.exp(1j * phaseArr1)
  #print 'Error : ', np.sum((newArr1 - dftArr1)**2)
  #print 'Error : ', np.sum((newArr2 - dftArr2)**2)
  #
  idftArr1 = idft_2d_from_1d(newArr1)
  idftArr2 = idft_2d_from_1d(newArr2)
  idftArr1 = np.round(idftArr1.real)
  idftArr2 = np.round(idftArr2.real)
  #
  dftArrAbs1 = np.abs(fftshift(dftArr1))
  dftArrAbs2 = np.abs(fftshift(dftArr2))
  #
  plt.figure()
  plt.subplot(121)
  plt.title(file1)
  plt.imshow(imgarr1, cmap=cm.gray)
  #plt.colorbar()
  plt.subplot(122)
  plt.title(file2)
  plt.imshow(imgarr2, cmap=cm.gray)
  #plt.colorbar()
  #
  plt.figure()
  plt.subplot(121)
  plt.title('DFT of ' + file1)
  plt.imshow(np.int32(np.log(dftArrAbs1)), cmap=cm.gray)
  plt.colorbar()
  plt.subplot(122)
  plt.title('DFT of ' + file2)
  plt.imshow(np.int32(np.log(dftArrAbs2)), cmap=cm.gray)
  plt.colorbar()
  #
  plt.figure()
  plt.subplot(121)
  plt.title('IDFT1')
  plt.imshow(np.int32(idftArr1), cmap=cm.gray)
  #plt.colorbar()
  plt.subplot(122)
  plt.title('IDFT2')
  plt.imshow(np.int32(idftArr2), cmap=cm.gray)
  #plt.colorbar()

if __name__ == '__main__':
  from datetime import datetime
  print datetime.time(datetime.now())
  #f1('peppers.pgm')
  f2('Fourier.pgm', 'fourier_transform.pgm')
  print datetime.time(datetime.now())
  plt.show()
