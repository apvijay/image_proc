#!/usr/bin/env python
""" This module defines a class Img with basic image operations
  Module          : Img
  Created by      : Vijay
  Creation Date   : Jan 9, 2011
  Current version : 1.51
  -------
  Changes:
  -------
  1.0  | Jan 9, 2011  | Creation (translate, rotate, is_out_of_bound).
  1.1  | Jan 11, 2011 | Added bilinear_pix, __sub__, __add__ funcs. Added
       |              | angle argument in rotate() to support both radian
       |              | and degree. Modified translate() based on 
       |              | bilinear_pix().
  1.2  | Jan 12, 2011 | Added create_kernel, conv, append_zeros funcs.
  1.3  | Jan 13, 2011 | Added scale().
  1.4  | Jan 19, 2011 | Added lsv_blur, create_sigma, 
       |              | create_gauss_kernel, find_gauss_support and
       |              | find_support funcs. Removed create_kernel().
  1.5  | Feb 02, 2011 | Added mod_lap and sum_mod_lap funcs.
  1.51 | Feb 04, 2011 | In conv(), if kernel is a single point, return
       |              | scaled image immediately.
  1.6  | Feb 10, 2011 | Added dft_2d_from_1d, idft_2d_from_1d funcs.
  """
from numpy import *

class Img:
  def __init__(self, *val):
    """ Initialises the image with
      (a) zeros if row and col are given as arguments
      (b) given numpy array """
    if isinstance(val[0],int):
      self.pix = zeros((val[0],val[1]),dtype=float32)
      self.row = val[0]
      self.col = val[1]
    if isinstance(val[0],ndarray):
      self.pix = float32(val[0])
      (self.row,self.col) = shape(val[0])
    self.grayLevel = 256
  
  def set_gray_level(self, val):
    self.grayLevel = val
  
  def translate(self, tx, ty, type='default'):
    """ Translates the image by (tx,ty).
    Options:
      type can be specified as 'bilinear'"""
    imgt = Img(self.row,self.col)
    for yt in range(self.row):
      for xt in range(self.col):
        xs = xt - tx
        ys = yt - ty
        if type == 'default':
          xs = round(xs)
          ys = round(ys)
          if not self.is_out_of_bound(xs,ys):
            imgt.pix[yt][xt] = self.pix[ys][xs]
        if type == 'bilinear':
          imgt.pix[yt][xt] = self.bilinear_pix(xs,ys)
    return imgt
  
  def bilinear_pix(self, x, y):
    """ Computes a bilinear interpolated value of the given pixel
    position (integer/non) using the surrounding four neighbour 
    pixels with appropriate weightage """
    x1 = floor(x)
    y1 = floor(y)
    dx = x - x1
    dy = y - y1
    pix00 = 0 if self.is_out_of_bound(x1,y1) else self.pix[y1][x1]
    pix01 = 0 if self.is_out_of_bound(x1,y1+1) else self.pix[y1+1][x1]
    pix10 = 0 if self.is_out_of_bound(x1+1,y1) else self.pix[y1][x1+1]
    pix11 = 0 if self.is_out_of_bound(x1+1,y1+1) else self.pix[y1+1][x1+1]
    pix = (1-dx)*(1-dy)*pix00 + (1-dx)*dy*pix01 +\
          dx*(1-dy)*pix10 + dx*dy*pix11
    pix = floor(pix)
    return pix
  
  def rotate(self, theta=0, angle='degree'):
    """ Rotates the image by an angle theta degrees
    Options:
      angle='radian' or 'degree'(default) """
    imgt = Img(self.row, self.col)
    if angle == 'degree': t = theta*pi/180
    if angle == 'radian': t = theta
    A = mat([[cos(t), -sin(t)],[sin(t), cos(t)]])
    for yt in range(self.row):
      for xt in range(self.col):
        b = A*mat([[xt],[yt]])
        xs = floor(float(b[0]))
        ys = floor(float(b[1]))
        if not self.is_out_of_bound(xs,ys):
          imgt.pix[yt][xt] = self.pix[ys][xs]
    return imgt
  
  def is_out_of_bound(self, x, y):
    """ Returns 1 if the given pixel position is outside the image"""
    if x<0 or x>self.col-1 or y<0 or y>self.row-1:
      return 1
    else:
      return 0
  
  def __sub__(self, img2):
    """ Operator overloading for minus operator """
    imgt = Img(self.row, self.col)
    imgt.pix = self.pix - img2.pix
    return imgt
  
  def __add__(self, img2):
    """ Operator overloading for plus operator """
    imgt = Img(self.row, self.col)
    imgt.pix = self.pix + img2.pix
    return imgt
  
  def __abs__(self):
    """ Operator overloading for abs operator """
    imgt = Img(self.row, self.col)
    imgt.pix = abs(self.pix)
    return imgt
  
  def conv(self, kern):
    """ Convolves the image with kernel. Returns a new image 
    of class Img.
    Options:
      kern is a numpy array
    """
    if size(kern.shape) == 1:
      # Kernel is just a scaling operator.
      # Don't convolve. Return scaled image immediately.
      imgt = Img(self.pix)
      imgt.pix = imgt.pix * kern[0]
      return imgt
    w = shape(kern)[0] # Assume kernel is square
    s = (w-1)/2
    # imgt1 - zeros appended image
    # imgt2 - convolved image
    imgt1 = self.append_zeros(s,s)
    imgt2 = Img(self.row, self.col)
    # Move kernel over each pixel, perform weighted average 
    # and assign it to target image
    for i in arange(s, imgt1.row-s):
      for j in arange(s, imgt1.col-s):
        imgt2.pix[i-s][j-s] =\
        floor(sum(imgt1.pix[i-s:i+s+1][:,j-s:j+s+1] * kern))
    return imgt2
  
  def append_zeros(self, row, col):
    """ Append required number of zero-filled-rows and 
    zero-filled-columns to the image """
    imgt = Img(self.pix)
    # Append zeros to top and bottom of image
    zx = zeros((row, imgt.col))
    imgt.pix = append(zx, imgt.pix, axis=0)
    imgt.pix = append(imgt.pix, zx, axis=0)
    # Append zeros to left and right of image
    (imgt.row, imgt.col) = shape(imgt.pix)
    zy = zeros((imgt.row, col))
    imgt.pix = append(zy, imgt.pix, axis=1)
    imgt.pix = append(imgt.pix, zy, axis=1)
    (imgt.row, imgt.col) = shape(imgt.pix)
    return imgt
  
  def scale(self, factor, type='default'):
    """ Scale the image by the given factor. 'factor' is a two-element 
    tuple for x and y factors.
    factor > 1 => zoom-in
    factor < 1 => zoom-out
    Options:
      type can be specified as 'bilinear'"""
    imgt = Img(self.row, self.col)
    for yt in range(self.row):
      for xt in range(self.col):
        xs = xt / factor[0]
        ys = yt / factor[1]
        if type == 'default':
          xs = round(xs)
          ys = round(ys)
          if not self.is_out_of_bound(xs,ys):
            imgt.pix[yt][xt] = self.pix[ys][xs]
        if type == 'bilinear':
          imgt.pix[yt][xt] = self.bilinear_pix(xs,ys)
    return imgt
  
  def lsv_blur(self, sigma):
    """ Linear space variant blur """
    suppMat = find_support(sigma)
    maxSupp = suppMat.max()
    maxs = (maxSupp - 1) / 2
    imgt1 = self.append_zeros(maxs,maxs)
    imgt2 = Img(imgt1.row, imgt1.col)
    for i in arange(len(suppMat)):
      for j in arange(len(suppMat)):
        kern = create_gauss_kernel(suppMat[i][j], sigma[i][j])
        s = (suppMat[i][j] - 1) /2
        imgt2.pix[i+maxs-s:i+maxs+s+1][:,j+maxs-s:j+maxs+s+1] =\
        imgt2.pix[i+maxs-s:i+maxs+s+1][:,j+maxs-s:j+maxs+s+1] +\
        imgt1.pix[i+maxs][j+maxs] * kern
        #imgt1.pix[i+maxs-s:i+maxs+s+1][:,j+maxs-s:j+maxs+s+1] * kern
    imgt2.pix = imgt2.pix[maxs:maxs+self.row][:,maxs:maxs+self.col]
    imgt2.pix = floor(imgt2.pix)# / imgt2.pix.max() * 255
    imgt2.row = self.row
    imgt2.col = self.col
    return (suppMat, imgt2)
  
  def mod_lap(self):
    """ Modified Laplacian operator
    Returns a new image of class Img
    """
    kernx = array([[0,0,0],[1,-2,1],[0,0,0]])
    kerny = array([[0,1,0],[0,-2,0],[0,1,0]])
    dervx = self.conv(kernx)
    dervy = self.conv(kerny)
    modLapImg = abs(dervx) + abs(dervy)
    return modLapImg
  
  def sum_mod_lap(self, q):
    """ Sum modified Laplacian operator
    Returns a new image of class Img
    """
    kern = ones((2*q+1, 2*q+1))
    modLapImg = self.mod_lap()
    sumModLap = modLapImg.conv(kern)
    return sumModLap
  
  def bilateral_filter(self, sigmaD, sigmaR, filtType='bilateral'):
    """ Bilateral (domain and range) filtering
    """
    # Create Gaussian kernel (domain filter)
    supp = find_gauss_support(sigmaD)
    kernD = zeros((supp,supp))
    supp_range = float64(arange(-(supp-1)/2, (supp-1)/2 + 1))
    for i in supp_range:
      for j in supp_range:
        kernD[i+(supp-1)/2][j+(supp-1)/2] = \
        exp(-(i**2 + j**2)/(2*sigmaD**2))
    if filtType == 'range': kernD = ones((supp,supp))
    s = (supp-1)/2
    imgt1 = self.append_zeros(s,s) # zeros appended image
    imgt2 = Img(self.row, self.col) # convolved image (output)
    # Move kernel over each pixel and filter
    for i in arange(s, imgt1.row-s):
      for j in arange(s, imgt1.col-s):
        # Create range kernel
        A = (imgt1.pix[i-s:i+s+1][:,j-s:j+s+1] - 
             imgt1.pix[i][j]*ones((supp,supp)))**2
        kernR = exp(-A / (2 * sigmaR**2))
        if filtType == 'domain': kernR = ones((supp,supp))
        # Combine domain and range kernels
        kern = kernD * kernR
        # Normalise resulting kernel
        kern = kern / sum(kern)
        # Perform filtering
        imgt2.pix[i-s][j-s] =\
        floor(sum(imgt1.pix[i-s:i+s+1][:,j-s:j+s+1] * kern))
    return imgt2
  
  def calc_hist(self):
    """ Returns an array of L values, corresponding to gray levels
    0 to L-1, containing their count of their occurences
    """
    hist = zeros(self.grayLevel)
    for grayVal in arange(self.grayLevel): # 0 to L-1
      hist[grayVal] = sum(self.pix == grayVal)
    #hist = hist / (self.row * self.col)
    return hist

  def otsu_thresh():
    """ Threshold the image using Otsu thresholding.
    Returns binary image (actually grayscale, 0 and self.grayLevel-1)
    and the threshold used, as a tuple
    """
    from copy import deepcopy
    L = self.grayLevel
    hist = img.calc_hist(L)
    muT = sum(arange(L) * hist[arange(L)]) / sum(hist[arange(L)])
    def calc_sigmaBsq(gLevel):
      N1 = sum(hist[0:gLevel+1])
      N2 = sum(hist[gLevel+1:L])
      mu1 = sum(arange(0,gLevel+1) * hist[arange(0,gLevel+1)]) / N1
      mu2 = sum(arange(gLevel+1,L) * hist[arange(gLevel+1,L)]) / N2
      sigmaBsq = ((mu1 - muT)**2 * N1 + (mu2 - muT)**2 * N2) / (N1+N2)
      return sigmaBsq
    sigmaBsq = zeros(L)
    for gLevel in arange(1,L-1):
      sigmaBsq[gLevel] = calc_sigmaBsq(gLevel)
    sigmaBsq[isnan(sigmaBsq)] = 0
    thresh = sigmaBsq.argmax(axis=0)
    imgt = deepcopy(self)
    imgt.pix[where(imgt.pix > thresh)] = L-1
    imgt.pix[where(imgt.pix <= thresh)] = 0
    return imgt, thresh

  def add_salt_pepper(self, thresh):
    """ Add salt and pepper noise to the image. Returns a new
    image
    """
    from copy import deepcopy
    from numpy.random import rand
    imgt = deepcopy(self)
    for yt in range(imgt.row):
      for xt in range(imgt.col):
        # Add noise based on x
        x = rand(1)
        if x >= thresh:
          # Add salt or pepper based on y
          y = rand(1)
          imgt.pix[yt][xt] = self.grayLevel - 1 if y > 0.5 else 0
    return imgt

  def median_filter(self, kern):
    """ Apply median filter to the image with a kernel of size
    support x support. Return a new image
    """
    w = shape(kern)[0] # Assume kernel is square
    s = (w-1)/2
    # imgt1 - zeros appended image
    # imgt2 - filtered image
    imgt1 = self.append_zeros(s,s)
    imgt2 = Img(self.row, self.col)
    # Move kernel over each pixel which are 255 or 0,
    # perform weighted median
    # and assign it to target image
    for i in arange(s, imgt1.row-s):
      for j in arange(s, imgt1.col-s):
        if True:
        #if self.pix[i-s][j-s] == 255 or self.pix[i-s][j-s] == 0:
          imgt2.pix[i-s][j-s] =\
          median(weighted_window(imgt1.pix[i-s:i+s+1][:,j-s:j+s+1],
                 kern))
          imgt2.pix[i-s][j-s] = floor(imgt2.pix[i-s][j-s])
        else:
          imgt2.pix[i-s][j-s] = self.pix[i-s][j-s]
    return imgt2

  def modify_hist(self, imgref=None):
    """ Modify the image's histogram based on imgt image
    """
    from copy import deepcopy
    hist1 = self.calc_hist() / (self.row * self.col)
    if imgref:
      hist2 = imgref.calc_hist() / (imgref.row * imgref.col)
    else:
      # if ref image is not given, do hist equalization
      hist2 = ones(self.grayLevel) / self.grayLevel
    cumul1 = cumsum(hist1)
    cumul2 = cumsum(hist2)
    newLevel = arange(self.grayLevel)
    # Create an intensity map based on cdfs
    for gLevel in arange(len(cumul1)):
      # get index of nearest greater value
      newLevel[gLevel] = argmin(abs(cumul2 - cumul1[gLevel]))
      if cumul2[newLevel[gLevel]] < cumul1[gLevel]:
        newLevel[gLevel] = newLevel[gLevel] + 1
    imgt1 = deepcopy(self)
    # Map intensity values
    for yt in range(imgt1.row):
      for xt in range(imgt1.col):
        imgt1.pix[yt][xt] = newLevel[self.pix[yt][xt]]
    return imgt1
  
  def hist_eq(self):
    return self.modify_hist()
  
# End of class
def weighted_window(x,r):
  """ Weight (repeat) each element in x based on r and return
  an array
  """
  # convert to 1d array (one row)
  x1 = x.reshape(-1)
  r1 = r.reshape(-1)
  out = array([])
  for ind, elem in enumerate(x1):
    # [2] * 3 outputs [2,2,2]
    out = append(out, [elem] * r1[ind])
  return out

def plot_arr(arr, plotTitle):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  plt.figure()
  plt.plot(arr)
  plt.title(plotTitle)

def plot_hist(img, plotTitle):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  imgarray = img.calc_hist()
  plt.figure()
  plt.plot(imgarray)
  plt.title(plotTitle)

def stem_arr(arr, x, plotTitle):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  plt.figure()
  plt.stem(x, arr)
  plt.title(plotTitle)

def stem_hist(img, plotTitle):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  imgarray = img.calc_hist()
  plt.figure()
  plt.stem(arange(img.grayLevel), imgarray)
  plt.title(plotTitle)


def plot_fig(imgarr, plotTitle):
  import matplotlib.pyplot as plt
  from matplotlib import cm
  plt.figure()
  plt.imshow(imgarr, cmap=cm.gray)
  plt.title(plotTitle)

def dft_2d_from_1d(arr):
  """ Compute 2d DFT from 1d DFT. First, compute 1d DFT of every rows. 
  Then, compute 1d DFT of every column of resulting matrix"""
  from scipy.fftpack import fft
  r,c = arr.shape
  dftArr = zeros((r,c), dtype=complex)
  dftArr1 = zeros((r,c), dtype=complex).T
  for ind, row in enumerate(arr):
    dftArr[ind] = fft(row)
  for ind, col in enumerate(dftArr.T):
    dftArr1[ind] = fft(col)
  return dftArr1.T

def idft_2d_from_1d(arr):
  """ Compute 2d IDFT from 1d IDFT. First, compute 1d IDFT of 
  every rows. Then, compute 1d IDFT of every column of resulting 
  matrix"""
  from scipy.fftpack import ifft
  r,c = arr.shape
  idftArr = zeros((r,c), dtype=complex)
  idftArr1 = zeros((r,c), dtype=complex).T
  for ind, row in enumerate(arr):
    idftArr[ind] = ifft(row)
  for ind, col in enumerate(idftArr.T):
    idftArr1[ind] = ifft(col)
  return idftArr1.T


def find_support(sigma):
  """ Find the kernel support for Gaussian blur with sigma matrix """
  r,c = shape(sigma)
  s = zeros((r,c))
  for i in arange(r):
    for j in arange(c):
      s[i][j] = find_gauss_support(sigma[i][j]) 
  return s

def find_gauss_support(sigma):
  """ Calculates the kernel support for Gaussian blur
  support = smallest odd number greater or equal to 6*sigma+1
  """
  if sigma == 0: return 1
  supp = ceil(6*sigma + 1)
  if supp % 2 == 0: supp = supp + 1
  return supp

def create_gauss_kernel(supp, sigma):
  """ Creates a Gaussian kernel with support based on support
  kernel size = support x support"""
  kern = zeros((supp,supp))
  supp_range = float64(arange(-(supp-1)/2, (supp-1)/2 + 1))
  for i in supp_range:
    for j in supp_range:
      kern[i+(supp-1)/2][j+(supp-1)/2] = \
      exp(-(i**2 + j**2)/(2*sigma**2))/(2*pi*sigma**2)
  kern = kern / sum(kern)
  return kern

def create_sigma(a, b, s):
  """ Creates a Gaussian distribution for sigma of size NxN """
  (M,N) = s
  range1 = float64(arange(M))
  range2 = float64(arange(N))
  kern = zeros((M,N))
  for i in range1:
    for j in range2:
      kern[i][j] = \
      exp(-((i-M/2)**2 + (j-N/2)**2)/b)*a
  return kern
