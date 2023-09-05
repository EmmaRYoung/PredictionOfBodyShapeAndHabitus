# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:05:49 2022

@author: emmay
"""

import math
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Orient(SJ, KJ, xyzKinect):
    #Kabsch algorithm!
    #find rotation of "A" (KJ) into "B" (SJ)
    
    KR = [12, 22, 18, 5]
    SR = [17, 2, 1, 16]
    
    #mean center Kinect, SJ is already basically mean centered for our purposes.
    KJ_c = np.empty(KJ.shape, dtype = KJ.dtype)
    xyzKinect_c = np.empty(xyzKinect.shape, dtype = xyzKinect.dtype)
    for i in range(3):
        KJ_c[:,i] = KJ[:,i] - np.mean(KJ[:,i])
        xyzKinect_c[:,i] = xyzKinect[:,i] - np.mean(KJ[:,i]) #Kinect PC is centered with mean of KJ
        
    A = KJ_c[KR,:]
    A_centr = [np.mean(KJ_c[:,0]), np.mean(KJ_c[:,1]), np.mean(KJ_c[:,2])]
    
    B = SJ[SR,:]
    B_centr = [np.mean(SJ[:,0]), np.mean(SJ[:,1]), np.mean(SJ[:,2])]
    
    #compute covariance matrix
    H = np.matmul(B.T,A)
    
    #compute SVD of H
    #the numpy library returns, U, the diagonal of S, and V^H (conjugate transpose)
    [U, Sdiag, VH] = LA.svd(H)
    #reconstruct S and also get V
    S = np.zeros(H.shape)
    np.fill_diagonal(S, Sdiag)
    V = VH.T
    
    
    #check for special reflection case
    d = np.sign(LA.det(np.matmul(V,U.T)))
    matrix = np.identity(3)
    matrix[2,2] = d
    
    #rotation matrix
    R = np.matmul(V,np.matmul(matrix,U.T))
    
    #translation
    T = np.subtract(B_centr,A_centr)
    
    #calculate aligned Kinect joints and point cloud
    KJ_aligned = np.matmul(KJ_c,R) - T
    xyzKinect_aligned = np.matmul(xyzKinect_c,R) - T
    
    return KJ_aligned, xyzKinect_aligned

