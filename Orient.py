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



def Orient1(AJ,KJ,xyzKinect):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    '''
    ax.scatter(AJ[:,0], AJ[:,1], AJ[:,2])
    ax.scatter(KJ[:,0], KJ[:,1], KJ[:,2])
    ax.scatter(xyzKinect[:,0], xyzKinect[:,1], xyzKinect[:,2])
    ax.view_init(-90,0)
    ax.set_title('Unchanged original KJ and K_PC')
    plt.show()
    '''
    
    #this is not a smart way of doing this
    #change so works for arbitrary joints
    rz = R.from_euler('z', 180, degrees=True)
    ry = R.from_euler('y', 180, degrees=True)
    print("Rz")
    print(rz)
    
    print("Ry")
    print(ry)
    
    KJ = KJ.dot(rz.as_matrix())
    KJ = KJ.dot(ry.as_matrix())
    
    xyzKinect = xyzKinect.dot(rz.as_matrix())
    xyzKinect = xyzKinect.dot(ry.as_matrix())
    
    '''
    ax.scatter(AJ[:,0], AJ[:,1], AJ[:,2])
    ax.scatter(KJ[:,0], KJ[:,1], KJ[:,2])
    ax.scatter(xyzKinect[:,0], xyzKinect[:,1], xyzKinect[:,2])
    ax.view_init(-90,0)
    ax.set_title('First Rotation')
    plt.show()
    '''
    
    AJ_sm = np.zeros(3)
    KJ_sm = np.zeros(3)
    
    AJ_sm[0] = (AJ[17,0] + AJ[16,0])/2
    AJ_sm[1] = (AJ[17,1] + AJ[16,1])/2
    AJ_sm[2] = (AJ[17,2] + AJ[16,2])/2
    
    #define midpoint between kinect shoulders
    KJ_sm[0] = (KJ[5,0] + KJ[12,0])/2
    KJ_sm[1] = (KJ[5,1] + KJ[12,1])/2
    KJ_sm[2] = (KJ[5,2] + KJ[12,2])/2
    
    #move Kinect Skeleton to SMPL
    xdiff = AJ_sm[0] - KJ_sm[0]
    ydiff = AJ_sm[1] - KJ_sm[1]
    zdiff = AJ_sm[2] - KJ_sm[2]
    
    KJ = np.hstack((KJ,np.ones((len(KJ),1))))
    xyzKinect = np.hstack((xyzKinect,np.ones((len(xyzKinect),1))))
    
    
    T1 = np.identity(4)
    T1[0,3] = xdiff
    T1[1,3] = ydiff
    T1[2,3] = zdiff
    
    KJ = (T1.dot(KJ.T)).T
    xyzKinect = (T1.dot(xyzKinect.T)).T
    
    '''
    ax.scatter(AJ[:,0], AJ[:,1], AJ[:,2])
    ax.scatter(KJ[:,0], KJ[:,1], KJ[:,2])
    ax.scatter(xyzKinect[:,0], xyzKinect[:,1], xyzKinect[:,2])
    ax.view_init(-90,0)
    ax.set_title('First Translation')
    plt.show()
    '''
    
    #Move both to origin for rotation matrice to be applied to Kinect skeleton
    #recalc mid of shoulders for kinect
    KJ_sm = np.zeros(3)
    xdiff = (KJ[5,0] + KJ[12,0])/2
    ydiff = (KJ[5,1] + KJ[12,1])/2
    zdiff = (KJ[5,2] + KJ[12,2])/2
    
    T2 = np.identity(4)
    T2[0,3] = -xdiff
    T2[1,3] = -ydiff
    T2[2,3] = -zdiff
    
    AJ1 = np.hstack((AJ,np.ones((len(AJ),1))))
    AJ1 = (T2.dot(AJ1.T)).T
    
    KJ = (T2.dot(KJ.T)).T
    xyzKinect = (T2.dot(xyzKinect.T)).T
    
    ##Rotate
    #recalc mid of shoulders for defining vectors later
    sMid = np.zeros(3)
    sMid[0] = (KJ[5,0] + KJ[12,0])/2
    sMid[1] = (KJ[5,1] + KJ[12,1])/2
    sMid[2] = (KJ[5,2] + KJ[12,2])/2
    
    #define vectors to find angles
    #midpoint of hips
    #smpl
    AJ_hm = np.zeros(3) #wrong?
    AJ_hm[0] = (AJ1[2,0] + AJ1[1,0])/2
    AJ_hm[1] = (AJ1[2,1] + AJ1[1,1])/2
    AJ_hm[2] = (AJ1[2,2] + AJ1[1,2])/2
    
    
    #kinect
    KJ_hm = KJ[0,0:3] #hip mid for kinect is just the first joint in the skeleton
    
    
    #find angle in x
    Avect = sMid - AJ_hm #wrong
    Kvect = sMid - KJ_hm #wrong
    
    
    dot_prod = Avect.dot(Kvect)
    Anorm = LA.norm(Avect)
    Knorm = LA.norm(Kvect)
    
    angle = math.degrees((math.acos(dot_prod/(Anorm*Knorm))))
    
    
    rx = R.from_euler('x', angle, degrees=True)
    
    KJ = KJ[:,0:3].dot(rx.as_matrix())
    xyzKinect = xyzKinect[:,0:3].dot(rx.as_matrix())
    
    #find angle in z
    #create vector from shoulder mid to left hip
    KLhip = KJ[22,0:3]
    ALhip = AJ1[2,0:3]
    
    
    Avect = sMid - ALhip
    Kvect = sMid - KLhip
    dot_prod = Avect.dot(Kvect)
    Anorm = LA.norm(Avect)
    Knorm = LA.norm(Kvect)
    angle = math.degrees((math.acos(dot_prod/(Anorm*Knorm))))
    

    rz = R.from_euler('z', -angle, degrees=True)
    KJ = KJ[:,0:3].dot(rz.as_matrix())
    xyzKinect = xyzKinect[:,0:3].dot(rz.as_matrix())
    
    #revert back to positions before rotation matrices
    KJ = np.hstack((KJ,np.ones((len(KJ),1))))
    KJ = (LA.inv(T2).dot(KJ.T)).T
    
    xyzKinect = np.hstack((xyzKinect,np.ones((len(xyzKinect),1))))
    xyzKinect = (LA.inv(T2).dot(xyzKinect.T)).T
    
    #remove column of ones
    KJ = KJ[:,0:3]
    xyzKinect = xyzKinect[:,0:3]
    
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(AJ[:,0], AJ[:,1], AJ[:,2])
    #ax.scatter(KJ[:,0], KJ[:,1], KJ[:,2])
        
    return KJ, xyzKinect