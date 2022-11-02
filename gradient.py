# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:02:47 2021

@author: emmay
"""

import torch
import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import numpy.linalg as LA
from ObjFcn import f_1 
from ObjFcn import f_2
from ObjFcn import f_3

#bm_path = '../body_models/smplh/neutral/model.npz'
#dmpl_path = '../body_models/dmpls/neutral/model.npz'
#num_betas = 10 # number of body parameters
#num_dmpls = 8



#create gradient
def gradient_1(betas, pose_body, KJ,bm):
    betas = np.reshape(betas,(1,10)) #unchanged original betas     
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    joints = c2c(body.Jtr[0])
    AJ = np.asarray(joints)
    
    L = 20
    g = torch.Tensor(np.zeros((10,1)))
    #delta = 0.0001
    delta = 0.0001 #changed to avoid noise in parameters
    betas_initial = np.reshape(betas,(10,1)) #unchanged original betas 
    
    
    for i in range(L):
        #create z (in unit sphere)
        a = np.random.randn(10,1)
        z = torch.Tensor(a/LA.norm(a))
        
        betas = betas_initial + delta*z 
        
        betas = np.reshape(betas,(1,10)) #reshaped to go into smpl body scripts
        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        joints = c2c(body.Jtr[0])
        AJ_p = np.asarray(joints)
        
        #compile terms into the gradient
        g = g + ((f_1(AJ_p,KJ) - f_1(AJ,KJ))/delta)*z
    
    g = g/L #average
    
    return g


def gradient_2(betas, pose_body, KJ,bm):
    pose_body = np.reshape(pose_body,(1,63)) #unchanged original betas     
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    joints = c2c(body.Jtr[0])
    AJ = np.asarray(joints)
    
    L = 20
    g = torch.Tensor(np.zeros((63,1)))
    #delta = 0.0001
    delta = 0.0001 #changed to avoid noise in parameters
    pose_initial = np.reshape(pose_body,(63,1)) #unchanged original pose 
    
    
    for i in range(L):
        #create z (in unit sphere)
        a = np.random.randn(63,1)
        z = torch.Tensor(a/LA.norm(a))
        
        pose_body = pose_initial + delta*z 
        
        pose_body = np.reshape(pose_body,(1,63)) #reshaped to go into smpl body scripts
        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        joints = c2c(body.Jtr[0])
        AJ_p = np.asarray(joints)
        
        #compile terms into the gradient
        g = g + ((f_2(AJ_p,KJ) - f_2(AJ,KJ))/delta)*z
    
    g = g/L #average
    
    return g

def gradient_3(betas, pose_body, xyzKinect, bm, Indices):
    #print("grad")
    betas = np.reshape(betas,(1,10)) #reshaped to go into smpl body scripts
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])   
    
    #pose_body = np.reshape(pose_body,(63,1))
    #betas = np.reshape(betas,(10,1))
    
    
    L = 5
    g = torch.Tensor(np.zeros((10,1)))
    #delta = 0.0001
    delta = 0.0001 #changed to avoid noise in parameters
    
    
    betas_initial = np.reshape(betas,(10,1)) #unchanged original betas 
    
    for i in range(L):
        #create z (in unit sphere)
        a = np.random.randn(10,1)
        z = torch.Tensor(a/LA.norm(a))
        
        betas = betas_initial + delta*z 
        
        betas = np.reshape(betas,(1,10)) #reshaped to go into smpl body scripts
        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL_p = c2c(body.v[0])   
        
        #compile terms into the gradient
        g = g + ((f_3(xyzSMPL_p, xyzKinect, Indices) - f_3(xyzSMPL, xyzKinect, Indices))/delta)*z
    
    g = g/L #average
    
    return g

def gradient_4(betas, pose_body, xyzKinect, bm, Indices):
    #print("grad")
    pose_body = np.reshape(pose_body,(1,63)) #reshaped to go into smpl body scripts
    betas = np.reshape(betas,(1,10))
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])   
    
    #pose_body = np.reshape(pose_body,(63,1))
    #betas = np.reshape(betas,(10,1))
    
    
    L = 5
    g = torch.Tensor(np.zeros((63,1)))
    #delta = 0.0001
    delta = 0.0001 #changed to avoid noise in parameters
    
    
    pose_initial = np.reshape(pose_body,(63,1)) #unchanged original betas 
    
    for i in range(L):
        #create z (in unit sphere)
        a = np.random.randn(63,1)
        z = torch.Tensor(a/LA.norm(a))
        
        pose_body = pose_initial + delta*z 
        
        pose_body = np.reshape(pose_body,(1,63)) #reshaped to go into smpl body scripts
        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL_p = c2c(body.v[0])   
        
        #compile terms into the gradient
        g = g + ((f_3(xyzSMPL_p,xyzKinect,Indices) - f_3(xyzSMPL,xyzKinect,Indices))/delta)*z
    
    g = g/L #average
    
    return g