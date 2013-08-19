#!/usr/bin/env python
""" 
  Assignment : 6
  Created by : Vijay
  Date       : Feb 20, 2011
  Topics     : Singular Value Decomposition and demonstration of
               matrix compression by dropping insignificant singular values
"""
import numpy as np
from numpy import linalg as LA
from numpy import arange, size, shape, array, abs

g = np.matrix([
[255,255,255,255,255,255,255,255],
[255,255,255,100,100,100,255,255],
[255,255,100,150,150,150,100,255],
[255,255,255,255,200,255,255,255],
[255,255,255,255,150,255,255,255],
[255,255,255,100,100,100,255,255],
[255,255,255,255, 50,255,255,255],
[ 50, 50, 50, 50,255,255,255,255.],
])

def f1():
#  g = np.matrix([[4,1,1,1],[0,16,0,9],[0,0,9,1],[0,0,0,4.]])
  sigma, A, B = compute_svd(g) # g = A sigma B^T (or) B g = A sigma
  print 'SVD'
  print 'given g:\n', g.round(2)
  print 'A:\n', A.round(2)
  print 'B:\n', B.T.round(2)
  print 'sigma:\n', np.diag(sigma.round(2))
  print 'g = A sigma B^T:\n', (A * np.diag(sigma) * B.T).round(2)
  print 'Check : ', np.dot(g, A[:,0] * B[:,0].T)
  #inb_svd(g)
  exit()
  for cnt in np.arange(size(sigma)+1):#-size(sigma)-1):
    print '\n'
    pickVals = set(arange(size(sigma)-cnt)) # set reduces as cnt increases
    sigma1 = sigma[[i for i in pickVals]] # pick significant sing vals
    sigma2 = sigma[[i for i in set(arange(size(sigma)))-pickVals]]
    gHat = compute_gHat(sigma1, A, B) # form gHat
    error1 = compute_error(g, gHat) # norm square error of difference
    error2 = np.sum(sigma2**2) # error computed using dropped evals
    cr = (1 - size(sigma1) * (shape(A)[0] + shape(B)[1]) / float(size(g)))
    print 'sigma1:\n', sigma1.round(2)
    print 'gHat:\n', gHat.round(2)
    print 'error1 = norm^2(g - gHat):\n', error1.round(2)
    print 'error2 = sum(dropped evals^2):\n', error2.round(2)
    print 'Compression: %.2f %%' % (cr*100)
  

def compute_svd(g):
  """ Compute SVD: g = A sigma B^T
  B contains evecs of g^Tg as columns
  A = g B sigma_inv (will contain zero columns if g is non-invertible)
  Zero columns of A are replaced by the evecs of gg^T for zero eval
  """
  gtg = g.T * g
  ggt = g * g.T
  w1, A1 = LA.eig(ggt)
  w2, B = LA.eig(gtg)
  sigma = np.sqrt(w2)
  A1 = A1[:,np.argsort(w1)[-1::-1]] # cols of A1 are evecs of ggt
  B = B[:,np.argsort(w2)[-1::-1]] # cols of B are evecs of gtg
  sigma = np.sort(sigma)[-1::-1]
  # From above, g may not equal to A1 sigma B^T
  # g = AsB^T => gB = As => A = g B s_inv; here A2 is A
  # A2 contains zero columns corresponding to zero evals
  A2 =  g * B * np.diag(1/sigma) 
  # Create new A formed by A2's non-zero eval columns and
  # A1's remaining columns
  A = np.append(A2[:,sigma.real.round(2) != 0],
                A1[:,sigma.real.round(2) == 0], axis = 1)
  #A = A2
  return sigma.real, A.real, B.real # floating point results in 0j part

def compute_svd2(g):
  """ Compute SVD: g = A sigma B^T
  """
  gtg = g.T * g
  ggt = g * g.T
  w1, A = LA.eig(ggt)
  w2, B = LA.eig(gtg)
  w1 = w1.real
  w2 = w2.real
  w1neg = np.where(w1 < 0)
  w2neg = np.where(w2 < 0)
  A = A.real.round(2)
  B = B.real.round(2)
  w1 = w1.round(2)
  w2 = w2.round(2)
  for ind in w1neg:
    A[:,ind] = -A[:,ind]
    w1[ind] = -w1[ind]
  #for ind in w2neg:
    #B[:,ind] = -B[:,ind]
    #w2[ind] = -w2[ind]
  sigma = np.sqrt(w2)
  A = A[:,np.argsort(w1)[-1::-1]] # cols of A1 are evecs of ggt
  B = B[:,np.argsort(w2)[-1::-1]] # cols of B are evecs of gtg
  sigma = np.sort(sigma)[-1::-1]
  print A * sigma * B.T
  return sigma.real, A.real, B.real # floating point results in 0j part

def compute_gHat(sig, A, B):
  gHat = np.zeros((size(A,0),size(B,0)))
  for i in arange(size(sig)): # sig contains significant evals only
    ai = A[:,i]
    bi = B[:,i]
    gHat = gHat + sig[i] * ai * bi.T
  return gHat
    
def compute_error(g, gHat):
  return np.sum((array(g) - array(gHat))**2)

def inb_svd(g):
  print '---\n Inbuilt svd'
  q1,q2,q3 = LA.svd(g)
  print 'A:\n', q1.round(2), '\nB:\n', q3.round(2), \
        '\nsigma:\n', q2.round(2)
  print q1 * np.diag(q2) * q3
  print '---\n'

if __name__ == '__main__':
  f1()
  #compute_svd2(g)
