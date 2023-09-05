# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:57:35 2021

@author: emmay
"""
import numpy.linalg as LA
import numpy as np
import torch
from sklearn.neighbors import KDTree
import csv


'''
#define objective function
def f_1(AJ,KJ): #A: Amass Joints, KR: Kinect Joints
    ATR = LA.norm(AJ[1] - AJ[4]) #Amass thigh right    
    ATL = LA.norm(AJ[2] - AJ[5]) #Amass thigh left

    ACR = LA.norm(AJ[4] - AJ[7]) #Amass calf right
    ACL = LA.norm(AJ[5] - AJ[8]) #Amass calf left

    AFR = LA.norm(AJ[7] - AJ[10]) #Amass foot right
    AFL = LA.norm(AJ[8] - AJ[11]) #Amass foot left


    KTR = LA.norm(KJ[22] - KJ[23]) #Kinect thigh right
    KTL = LA.norm(KJ[18] - KJ[19]) #Kinect thigh left

    KCR = LA.norm(KJ[23] - KJ[24]) #Kinect calf right
    KCL = LA.norm(KJ[19] - KJ[20]) #Kinect calf left

    KFR = LA.norm(KJ[24] - KJ[25]) #Kinect foot right
    KFL = LA.norm(KJ[20] - KJ[21]) #Kinect foot left

    #Arms
    ABR = LA.norm(AJ[16] - AJ[18]) #amass bicept right
    AF_R = LA.norm(AJ[18] - AJ[20]) #amass forarm right

    ABL = LA.norm(AJ[17] - AJ[19]) #amass bicept left
    AF_L = LA.norm(AJ[19] - AJ[21]) #amass forarm left


    KBR = LA.norm(KJ[12] - KJ[13]) #kinect bicept right
    KF_R = LA.norm(KJ[13] - KJ[14]) #kineft forarm right

    KBL = LA.norm(KJ[5] - KJ[6]) #kinect bicept left
    KF_L = LA.norm(KJ[6] - KJ[7]) #kinect forarm left

    #trunk
    #Amass
    #define midpoint between hips
    #mid = (AJ[2]+AJ[1])/2;
    #AT = LA.norm(mid[0] - AJ[12])
    
    #NEW: from Lhip to Lshoulder
    #18 - 3
    AT = LA.norm(AJ[17] - AJ[2])

    #kinect
    #from pelvis to neck
    #KT = LA.norm(KJ[0] - KJ[3])
    
    #NEW: from Lhip to Lshoulder
    #6 - 19
    KT = LA.norm(KJ[5] - KJ[18])

    #shoulders
    AS = LA.norm(AJ[16] - AJ[17])
    KS = LA.norm(KJ[5] - KJ[12])

    
    #hips
    AH = LA.norm(AJ[1] - AJ[2])
    KH = LA.norm(KJ[18] - KJ[22])
    

    #Create vector to pass to objective function
    #Keep same order
    ALength = []
    KLength = []

    ALength.extend([ATR, ATL, ACR, ACL, AFR, AFL, ABR, AF_R, ABL, AF_L, AT, AS, AH])
    KLength.extend([KTR, KTL, KCR, KCL, KFR, KFL, KBR, KF_R, KBL, KF_L, KT, KS, KH])

    value = []
    for i in range(len(ALength)):
        value.append((ALength[i] - KLength[i])**2)

    fval = np.sum(value)


    return fval    
'''

def f_2(AJ,KJ): #objective function for matching pose
    
    #make sure things are the right size and remove possible 4th column of ones
    try:
        AJ = np.delete(AJ,3,1)
    except:
        AJ = AJ
        
    try:
        KJ = np.delete(KJ,3,1)
    except:
        KJ = KJ
       
    #Create a list of distances between similar joints
   
    #ARMS
    #right
    Shoulder_R = LA.norm(AJ[16] - KJ[5])
    Elbow_R = LA.norm(AJ[18] - KJ[6])
    Wrist_R = LA.norm(AJ[20] - KJ[7])
    #Finger_R = LA.norm(AJ[27] - KJ[9])
    
    #left
    Shoulder_L = LA.norm(AJ[17] - KJ[12])
    Elbow_L = LA.norm(AJ[19] - KJ[13])
    Wrist_L = LA.norm(AJ[21] - KJ[14])
    #Finger_L = LA.norm(AJ[42] - KJ[16])
    
    #LEGS
    #right
    Knee_R = LA.norm(AJ[4] - KJ[19])
    Foot_R = LA.norm(AJ[7] - KJ[20])
    Toe_R = LA.norm(AJ[10] - KJ[21])
    
    #left
    Knee_L = LA.norm(AJ[5] - KJ[23])
    Foot_L = LA.norm(AJ[8] - KJ[24])
    Toe_L = LA.norm(AJ[11] - KJ[25])
    

    
    
    compDist = []

    compDist.extend([Shoulder_R, Elbow_R, Elbow_R, Wrist_R, Shoulder_L, Elbow_L, Elbow_L, Wrist_L, Knee_R, Foot_R, Toe_R, Knee_L, Foot_L, Toe_L])


    fval = np.sum(compDist)
    

    return fval

'''
def f_3(xyzSMPL1, xyzKinect, Indices, betas, TorsoBool):
    #nearest neighbor comparison of subject specific point cloud to SMPL mesh
    
    #include the norm of the shape parameters in the objective function with weights
    betas = np.asarray(betas)
    bweight = 1
    
    bnorm = bweight*LA.norm(betas)
        
    #pull out indices of interest   
    xyzSMPL = xyzSMPL1[Indices]
    
    #pull out indices of torso
    xyzSMPL_torso = xyzSMPL1[TorsoBool]
    tweight = 100
    
    
    tree = KDTree(xyzKinect)
    vertError = []
    dist, ind = tree.query(xyzSMPL, k=1)   
    
   
    dist_t, ind = tree.query(xyzSMPL_torso, k=1)
    dist_t = dist_t*tweight
    
    vertError = np.vstack((dist, dist_t))
    
    totalError = sum(vertError)

    
    
    return totalError
'''

def f_3(xyzSMPL1, xyzKinect, betas, pose_body, Indices, TorsoBool, ExtrBool):
    #nearest neighbor comparison of subject specific point cloud to SMPL mesh
    
    #?? maybe include the norm of the shape parameters in the objective function with weights
    #pull out indices of interest   
    
    
    xyzSMPL = xyzSMPL1[Indices]
    
    #pull out indices of torso
    
    xyzSMPL_torso = xyzSMPL1[TorsoBool]
    tweight = 100
    
    xyzSMPL_extr = xyzSMPL1[ExtrBool]
    eweight = 10
  
    vertError = []
    
    tree = KDTree(xyzKinect)
    
    #dist, ind = tree.query(xyzSMPL, k=10)   
    dist, ind = tree.query(xyzSMPL, k=1)  
    dist = np.mean(dist, axis = 1)
    #save nearest neighbors for graping
    '''
    np.savetxt("SMPLVerts_KNN_test.txt", xyzSMPL, fmt = '%f')
    np.savetxt("KinectPC_KNN_test.txt", xyzKinect, fmt = '%f')
    np.savetxt("KNN_INDS.txt", ind, fmt = '%f')
    '''
    #dist_t, ind = tree.query(xyzSMPL_torso, k=10)
    dist_t, ind = tree.query(xyzSMPL_torso, k=1)
    dist_t = dist_t*tweight
    dist_t = np.mean(dist_t, axis = 1)
    
    dist_e, ind = tree.query(xyzSMPL_extr, k=1)
    dist_e = dist_e*tweight
    dist_e = np.mean(dist_e, axis = 1)
   
    #vertError = dist
    vertError = np.hstack((dist, dist_t, dist_e))
    
    #pose_norm = LA.norm(pose_body)
    #pose_mult = 1000
    
    totalError = sum(vertError) #+ pose_norm*pose_mult
        
    return totalError