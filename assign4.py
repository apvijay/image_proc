#!/usr/bin/env python
""" 
  Assignment : 4
  Created by : Vijay
  Date       : Feb 6, 2011
  Topics     : Shape from focus
"""
import Img
import time
import numpy as np
from scipy import io
from numpy import log
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

debug = 0
colorMap = cm.copper
d = 50.50

def get_stack(filename):
  """ Read mat file contents to dict """
  stack = io.loadmat(filename) # Read mat file contents as dict
  stack = {key:stack[key] for key in stack.keys() if 'frame' in key}
  if 'numframes' in stack: stack.pop('numframes')
  # stack contains only the frames from this point
  return stack

def calc_sml(file1, file2, q):
  """ Calculates SML for file1 frames with q(defined as in class), 
  and store as mat file file2
  """
  stack = get_stack(file1)
  stackOut = dict() # Define output stack as dict
  for key in sorted(stack.keys()):
    print key
    imgOut = Img.Img(stack[key]).sum_mod_lap(q) # create a new Img class
    stackOut[key] = imgOut.pix # Add array to output stack
  io.savemat(file2, stackOut)

def get_foc_img(file1, file2):
  stackOrig = get_stack(file1)
  stack = get_stack(file2)
  row, col = stack['frame001'].shape
  focImg = np.zeros((row,col))
  dbar = np.zeros((row,col))
  for i in np.arange(row):
    for j in np.arange(col):
      pixStack = np.array([])
      # For this pixel, get values from all frames
      for key in sorted(stack.keys()): 
        pixStack.resize(pixStack.size + 1)
        pixStack[-1] = stack[key][i][j]
      # Find the maxima position
      (mp,m,mn) = find_max_pos(pixStack)
      if debug > 1: # Debug figure. Plots pixStack.
        figg = plt.figure()
        ax = figg.add_subplot(111)
        ax.hold(True)
        ax.plot(pixStack)
      if debug > 0: # Debug print. Prints pixStack.
        print i, j
        print  pixStack
        print mp, m, mn
        print pixStack[mp], pixStack[m], pixStack[mn]
      # Get the pixel val from the frame which has max SML
      reqKey = 'frame' + '0'*(3-len(str(m+1))) + str(m+1)
      focImg[i][j] = stackOrig[reqKey][i][j]
      if mp == m or mn == m:
        # Peak is a flat region. 
        dbar[i][j] = m * d
      else:
        # One peak with prev and next values.
        reqKeyp = 'frame' + '0'*(3-len(str(mp+1))) + str(mp+1)
        reqKeyn = 'frame' + '0'*(3-len(str(mn+1))) + str(mn+1)
        dmp = mp * d 
        dmn = mn * d
        dm = m * d
        Fmp = stack[reqKeyp][i][j]
        Fmn = stack[reqKeyn][i][j]
        Fm = stack[reqKey][i][j]
        dbar[i][j] = ((dmn**2 - dm**2) * (log(Fm) - log(Fmp)) -\
                      (dmp**2 - dm**2) * (log(Fm) - log(Fmn)) ) / \
                      (2*d * (2*log(Fm) - log(Fmn) - log(Fmp)))
  # Uncomment this if you want the reverse of shape
  # dbar = dbar.max() - dbar
  #plt.figure()
  #plt.imshow(focImg, cmap=cm.gray)
  #plt.title('Focused image' + ' (' + file2 + ')')
  #plt.figure()
  #plt.imshow(dbar, cmap=colorMap)
  #plt.title('Shape' + ' (' + file2 + ')')
  #plt.colorbar()
  show_3d_shape(dbar, file2)

def show_3d_shape(dbar, file2):
  x = np.arange(dbar.shape[0])
  y = np.arange(dbar.shape[1])
  X,Y = np.meshgrid(x,y)
  fig1 = plt.figure()
  fig1.suptitle('Shape 3d' + ' (' + file2 + ')')
  ax = fig1.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, dbar,
        rstride=2, cstride=2,
        linewidth=0, 
        antialiased=False,
        cmap=colorMap)

def show_animation():
  stack = get_stack('stack.mat')
  fig = plt.figure()
  ax = fig.add_subplot(111)
  def animate():
    tstart = time.time() # for profiling
    data = stack['frame001']
    im = plt.imshow(data, cmap=cm.gray)
    for key in sorted(stack.keys()):
      #print key
      data = stack[key]
      im.set_data(data)
      fig.canvas.draw() # redraw the canvas
    print 'FPS:', 100/(time.time()-tstart)
  win = fig.canvas.manager.window
  fig.canvas.manager.window.after(200, animate)
  plt.show()

def find_max_pos(arr):
  """ Returns the indices of (peak value, its previous value 
  and its next value). Always return three indices.
  """
  ind = find_max_ind(arr)
  if ind.size == 1:
    # Only one peak(no consecutive values).
    return ind[0]-1, ind[0], ind[0]+1
  else:
    # Flat peak region. Return same peak value thrice.
    # This condition is/must be checked in the parent function.
    return ind[ind.size/2], ind[ind.size/2], ind[ind.size/2]
  # # Ignore below
  # # Below is an algo for flat peak region. It takes centre peak 
  # # value, prev value and next values as before and after lower vals
  #if (ind.size-1) % 2 == 0:
  #  # Multiple peaks. Size is odd, i.e. last index is even.
  #  # Return middle value.
  #  return ind[0]-1, ind[ind.size/2], ind[-1]+1
  #else:
  #  # Multiple peaks. Last index is odd. Return middle value, 
  #  # closer to maximum of previous and next values.
  #  if arr[ind[0]-1] > arr[ind[-1]+1]:
  #    return ind[0]-1, ind[np.floor((ind.size-1)/2)], ind[-1]+1
  #  else:
  #    return ind[0]-1, ind[np.floor((ind.size-1)/2)+1], ind[-1]+1

def find_max_ind(arr):
  """ Return the indices of maximum peak value(consecutive indices, 
  if present). Returns an array.
  """
  ind = np.array([])
  for i in range(1,arr.size-1):
    if arr[i-1] < arr[i]: # Func is increasing
      if (ind.size == 0 or arr[i] > arr[ind[-1]]) \
         and arr[i+1] <= arr[i]: # Found a peak, gt current peak
        ind = np.array([i])
    elif ind.size > 0:
      if arr[i] == arr[ind[-1]] and arr[i-1] == arr[i]:
        # Function is constant, if this eq the peak, append
        ind.resize(ind.size + 1)
        ind[-1] = i
  return ind

# Following are test functions
def mod_lap_test():
  imgs = Img.Img(np.array([[1,2,3],[0,1,5],[6,3,9]]))
  imgt = imgs.mod_lap()
  print imgs.pix
  print imgt.pix

def sum_mod_lap_test():
  imgs = Img.Img(np.array([[1,2,3],[0,1,5],[6,3,9]]))
  imgt = imgs.sum_mod_lap(1)
  print imgs.pix
  print imgt.pix

def find_max_pos_test():
  arr = np.array([1,2,3,4,4,4,4,3,2,1])
  ind = find_max_pos(arr)
  print ind

def find_max_ind_test():
  arr = np.array([9,0,1,2,3,4,5,6,7,8,8,8])
  ind = find_max_ind(arr)
  print ind

if __name__ == '__main__':
  from datetime import datetime
  print datetime.time(datetime.now())
  #mod_lap_test()
  #sum_mod_lap_test()
  #find_max_pos_test()
  #find_max_ind_test()
  #show_animation()
  #calc_sml('stack.mat', 'stackOut.mat', 0)
  get_foc_img('stack.mat', 'stackOut0.mat')
  print datetime.time(datetime.now())
  get_foc_img('stack.mat', 'stackOut1.mat')
  print datetime.time(datetime.now())
  get_foc_img('stack.mat', 'stackOut2.mat')
  print datetime.time(datetime.now())
  get_foc_img('stack.mat', 'stackOut3.mat')
  print datetime.time(datetime.now())
  plt.show()
  
