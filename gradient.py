# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:02:47 2021

@author: emmay
"""

import torch
import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import numpy.linalg as LA
#from ObjFcn import f_1 
from ObjFcn import f_2
from ObjFcn import f_3
#from ObjFcn import f_all
#bm_path = '../body_models/smplh/neutral/model.npz'
#dmpl_path = '../body_models/dmpls/neutral/model.npz'
#num_betas = 10 # number of body parameters
#num_dmpls = 8


'''
#create gradient
def gradient_1(betas, pose_body, KJ,bm):
    betas = np.reshape(betas,(1,10)) #unchanged original betas     
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    joints = c2c(body.Jtr[0])
    AJ = np.asarray(joints)
    
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
        joints = c2c(body.Jtr[0])
        AJ_p = np.asarray(joints)
        
        #compile terms into the gradient
        g = g + ((f_1(AJ_p,KJ) - f_1(AJ,KJ))/delta)*z
    
    g = g/L #average
    
    return g
'''

def gradient_2(betas, pose_body, KJ,bm):
    pose_body = np.reshape(pose_body,(1,63)) #unchanged original betas     
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    joints = c2c(body.Jtr[0])
    AJ = np.asarray(joints)
    
    L = 36
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

def gradient_3(betas, pose_body, xyzKinect, bm, Indices, TorsoBool, ExtrBool):
    #print("grad")
    betas = np.reshape(betas,(1,10)) #reshaped to go into smpl body scripts
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])   
    
    #pose_body = np.reshape(pose_body,(63,1))
    #betas = np.reshape(betas,(10,1))
    
    
    L = 11
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
        term = ((f_3(xyzSMPL_p, xyzKinect, betas, pose_body, Indices, TorsoBool, ExtrBool) - f_3(xyzSMPL, xyzKinect, betas, pose_body, Indices, TorsoBool, ExtrBool))/delta)
        term = torch.tensor(term)
        g = g + term*z
    
    g = g/L #average
    g = g.float()

    return g

def gradient_4(betas, pose_body, xyzKinect, bm, Indices, TorsoBool, ExtrBool):
    #print("grad")
    pose_body = np.reshape(pose_body,(1,63)) #reshaped to go into smpl body scripts
    betas = np.reshape(betas,(1,10))
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])   
    
    #pose_body = np.reshape(pose_body,(63,1))
    #betas = np.reshape(betas,(10,1))
    
    
    L = 64
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
        term = ((f_3(xyzSMPL_p, xyzKinect, betas, pose_body, Indices, TorsoBool, ExtrBool) - f_3(xyzSMPL, xyzKinect, betas, pose_body, Indices, TorsoBool, ExtrBool))/delta)
        term = torch.tensor(term)
        g = g + term*z
    
    g = g/L #average
    g = g.float()
    
    return g

'''
def gradient_allvars(solution, xyzKinect, bm, Indices):
    #solution should be a row vector entering this 
    
    betas = solution[:,0:10]
    pose_body = solution[:,10:73]
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])   
    
    #pose_body = np.reshape(pose_body,(63,1))
    #betas = np.reshape(betas,(10,1))
    
    
    L = 74
    g = torch.Tensor(np.zeros((73,1)))
    #delta = 0.0001
    delta = 0.001 #changed to avoid noise in parameters
    
    
    solution_initial = np.reshape(solution,(73,1)) #unchanged original solution vector 
    
    for i in range(L):
        #create z (in unit sphere)
        a = np.random.randn(73,1)
        z = torch.Tensor(a/LA.norm(a))
        
        solution = solution_initial + delta*z 
        
        solution = np.reshape(solution,(1,73))
        betas = solution[:,0:10]
        pose_body = solution[:,10:73]
        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL_p = c2c(body.v[0])   
        
        #compile terms into the gradient
        g = g + ((f_3(xyzSMPL_p,xyzKinect,Indices) - f_3(xyzSMPL,xyzKinect,Indices))/delta)*z
    
    g = g/L #average
    
    return g

def approxGrad(solution, bm, index, Indices, TorsoBool, ExtrBool, xyzKinect):
    #index must divide gradient vector into:
    #vars that control betas: solution[:,0:10]
    #vars that control pose: solution[:,10:73]
    
    #get vertices at starting location
    solution = np.reshape(solution, (1,73))  
    betas = solution[:,0:10]
    
    pose_body = solution[:,10:73]
    body = bm(pose_body=pose_body, betas=betas)
    xyzSMPL = c2c(body.v[0])
    
    numVars = np.int32(np.sum(index))
  
    
    #indexBool = np.reshape(index.astype(dtype=bool), (73,1))
    g = np.zeros((73,1))
    g = torch.Tensor(g) #73x1
    
    L = numVars + 1
    
    
    delta = 0.001

    solution = np.reshape(solution, (73,1))    
    solution_initial = solution
    
    
    for i in range(L):
        a = np.random.randn(numVars,1)
        z = torch.Tensor(a/LA.norm(a))
        z_full = torch.Tensor(np.zeros((73,1)))

        #place z vector within z_full vecor at spots specified by the index
        count = 0
        for ind in range(len(index)):
            if index[ind] == 1:
                z_full[ind] = z[count] 
                count = count + 1
        
        
        #only perturb entries in solution that have a corresponding "1" in the index variable
        #this is taken care of by modifying the z_full to have zeros in all other areas of 
        solution = solution_initial + delta*z_full
        
        
        solution = np.reshape(solution,(1,73))
        
        
        betas = solution[:,0:10]
        pose_body = solution[:,10:73]
        
        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL_p = c2c(body.v[0])
        
        #compile terms into the gradient
        term = ((f_all(xyzSMPL_p, xyzKinect, Indices, TorsoBool, ExtrBool) - f_all(xyzSMPL,xyzKinect, Indices, TorsoBool, ExtrBool))/delta)
        term = torch.tensor(term)
        g = g + term*z_full
    
   
    #by default, entries in g that we aren't concerned with are 0 from the creation of the z_full vector    
    g = g/L
    return g
'''