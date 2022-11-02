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



#define objective function
def f_1(AJ,KJ): #objective function for matching body shape with joints
    #A: SMPL Joints, KR: Kinect Joints
    ATR = LA.norm(AJ[1] - AJ[4]) #SMPL thigh right    
    ATL = LA.norm(AJ[2] - AJ[5]) #SMPL thigh left

    ACR = LA.norm(AJ[4] - AJ[7]) #SMPL calf right
    ACL = LA.norm(AJ[5] - AJ[8]) #SMPL calf left

    AFR = LA.norm(AJ[7] - AJ[10]) #SMPL foot right
    AFL = LA.norm(AJ[8] - AJ[11]) #SMPL foot left


    KTR = LA.norm(KJ[22] - KJ[23]) #Kinect thigh right
    KTL = LA.norm(KJ[18] - KJ[19]) #Kinect thigh left

    KCR = LA.norm(KJ[23] - KJ[24]) #Kinect calf right
    KCL = LA.norm(KJ[19] - KJ[20]) #Kinect calf left

    KFR = LA.norm(KJ[24] - KJ[25]) #Kinect foot right
    KFL = LA.norm(KJ[20] - KJ[21]) #Kinect foot left

    #Arms
    ABR = LA.norm(AJ[16] - AJ[18]) #SMPL bicept right
    AF_R = LA.norm(AJ[18] - AJ[20]) #SMPL forarm right

    ABL = LA.norm(AJ[17] - AJ[19]) #SMPL bicept left
    AF_L = LA.norm(AJ[19] - AJ[21]) #SMPL forarm left


    KBR = LA.norm(KJ[12] - KJ[13]) #kinect bicept right
    KF_R = LA.norm(KJ[13] - KJ[14]) #kineft forarm right

    KBL = LA.norm(KJ[5] - KJ[6]) #kinect bicept left
    KF_L = LA.norm(KJ[6] - KJ[7]) #kinect forarm left

    #trunk
    #SMPL
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

def f_2(AJ,KJ): #objective function for matching pose with skeletal joints
    
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


def f_3(xyzSMPL1, xyzKinect, Indices):
#nearest neighbor comparison
    
    
    #Pull out only vertices on the front and side. The kinect data doesn't include a small section on the back. 
    xyzSMPL = xyzSMPL1[Indices]
    
    
    tree = KDTree(xyzKinect)

    vertError = []
    for i in range(len(xyzSMPL)):
            point = np.asarray([xyzSMPL[i,:]])
            dist, ind = tree.query(point, k=1)
            vertError.append(np.mean(dist))

    totalError = sum(vertError)
    
    
    return totalError
