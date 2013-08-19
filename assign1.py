#!/usr/bin/env python
""" 
  Assignment : 1
  Created by : Vijay
  Date       : Jan 9, 2011
  Topics     : Translation, Rotation
"""
import Image
import pgm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def f1():
  (imgarr,w,h) = pgm.pgmread('lena.pgm')
  imgs = Image.Image(imgarr)
  imgt1 = imgs.translate(3.75,4.3)
  imgt2 = imgs.translate(3.75,4.3,'bilinear')
  plt.figure(1)
  plt.imshow(imgs.pix,cmap=cm.gray)
  plt.figure(2)
  plt.imshow(imgt1.pix,cmap=cm.gray)
  plt.figure(3)
  plt.imshow(imgt2.pix,cmap=cm.gray)
  plt.show()

def f2():
  (imgarr,w,h) = pgm.pgmread('pisa.pgm')
  imgs = Image.Image(imgarr)
  imgt1 = imgs.rotate(2.5)
  imgt2 = imgt1.rotate(-2.5)
  imgt3 = imgs - imgt2
  plt.figure(1)
  plt.imshow(imgs.pix,cmap=cm.gray)
  plt.figure(2)
  plt.imshow(imgt1.pix,cmap=cm.gray)
  plt.figure(3)
  plt.imshow(imgt2.pix,cmap=cm.gray)
  plt.figure(4)
  plt.imshow(imgt3.pix,cmap=cm.gray)
  plt.show()

if __name__ == '__main__':
#  f1()
  f2()
