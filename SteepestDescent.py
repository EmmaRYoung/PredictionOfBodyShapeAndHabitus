# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Fri Dec 17 13:43:50 2021

@author: emmay
"""
import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch

from ObjFcn import f_1 
from ObjFcn import f_2
from ObjFcn import f_3

from gradient import gradient_1
from gradient import gradient_2
from gradient import gradient_3
from gradient import gradient_4

from LineSearch import find_stepsize_approx_exact_1 as findstep1
from LineSearch import find_stepsize_approx_exact_2 as findstep2
from LineSearch import find_stepsize_approx_exact_3 as findstep3
from LineSearch import find_stepsize_approx_exact_4 as findstep4


import numpy.linalg as LA

#bm_path = '../body_models/smplh/neutral/model.npz'
#dmpl_path = '../body_models/dmpls/neutral/model.npz'
#num_betas = 10 # number of body parameters
#num_dmpls = 8
#comp_device = torch.device("cpu")


#Steepest Descent Algorithm
def SteepestDescent_1(betas_i, pose_body, KJ, N, bm):
    #AJMov = []
    #AV = []
    stepStore = []
    #stepStore_test = []

    betas = betas_i
    #betas_test = betas_i
    
    #store the norm of the gradient
    g_norm = []
    
    body = bm(pose_body=pose_body, betas=betas_i)#, root_orient=root_orient, trans=trans)
    AJ = np.asarray(c2c(body.Jtr[0]))
        
    #AJMov[0]['I'] = AJ
    #AV[0]['I'] = np.asarray(c2c(body.v[0]))
    
    history_b = []
    #history_b_test = []
    history_f = []
    #history_f_test = []
    
    #print(torch.Tensor.size(betas_i))
    #print(betas_i)
    #print(torch.Tensor.size(betas_i[0][None, :]))
    
    history_b = torch.cat([betas_i[0][None,:]], dim=1)#, axis = 1)
    #history_b_test = torch.cat([betas_i[0][None,:]], dim=1)
    history_f.append(f_1(AJ,KJ))
    #history_f_test.append(f(AJ,KJ))
    
    eps = 100
    tol = 0.01
    for i in range(N):
        print("Steepest Descent Iteration")
        print(i)
        
        g = gradient_1(betas,pose_body,KJ,bm)
        #g_test = gradient_test(betas_test,KJ)
        g_norm.append(LA.norm(g[:,0]))
        betas = np.reshape(betas,(1,10))
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        AJ = np.asarray(c2c(body.Jtr[0]))
        
        curval = f_1(AJ,KJ)
        betas = np.reshape(betas,(10,1))
        B = 2/(1+np.sqrt(5)) #from SARA FRIDOVICH-KEIL AND BENJAMIN RECHT paper
        T = 1000
        [step, fval] = findstep1(T, B, betas, pose_body, g, curval, bm, KJ)
       
        stepStore.append(step)
        #stepStore_test.append(step_test)
        
        betas = betas - step*g
        
        #body = bm(pose_body=pose_body, betas=betas_test)#, root_orient=root_orient, trans=trans)
        #AJ_test = np.asarray(c2c(body.Jtr[0]))
        #AJMov[i+1]['I'] = AJ
        #AV[i+1]['I'] = np.asarray(c2c(body.v[0]))
        if i == 0:
            #history_b = torch.cat((history_b[None,:], betas[None,:]), dim=1)#, axis = 1)
            #history_b_test = torch.cat((history_b_test[None,:], betas_test[None,:]), dim=1)#, axis = 1)
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
        else:
            #history_b = torch.cat((history_b, betas[None,:]), dim=1)
            #history_b_test = torch.cat((history_b_test, betas_test[None,:]), dim=1)#, axis = 1)
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
            
        #print(i)
        #print(f(AJ,KJ))
        #print(history_b)
        
    return betas, history_b, history_f, g, stepStore, g_norm


def SteepestDescent_2(betas, pose_initial, KJ, N, bm):
    
    stepStore = []

    pose_body = pose_initial
    
    
    #store the norm of the gradient
    g_norm = []
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    AJ = np.asarray(c2c(body.Jtr[0]))
        
    
    history_b = []
    history_f = []
  
    
    history_b = torch.cat([pose_body[0][None,:]], dim=1)#, axis = 1)
    #history_b_test = torch.cat([betas_i[0][None,:]], dim=1)
    history_f.append(f_2(AJ,KJ))
    #history_f_test.append(f(AJ,KJ))
    
    
    for i in range(N):
        print("Steepest Descent Iteration")
        print(i)
        
        g = gradient_2(betas,pose_body,KJ,bm)
        #g_test = gradient_test(betas_test,KJ)
        g_norm.append(LA.norm(g[:,0]))
        #get current function value
        pose_body = np.reshape(pose_body,(1,63))
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        AJ = np.asarray(c2c(body.Jtr[0]))
        
        

        #print(np.transpose(gtemp)*gtemp)
       
        curval = f_2(AJ,KJ)
        B = 2/(1 + np.sqrt(5)) #from SARA FRIDOVICH-KEIL AND BENJAMIN RECHT paper
        pose_body = np.reshape(pose_body,(63,1))
        T = 1
        [step, fval] = findstep2(T, B, betas, pose_body, g, curval, bm, KJ)
       
        stepStore.append(step)
        #stepStore_test.append(step_test)
        
        pose_body = pose_body - step*g
       
        
        #betas_test = np.reshape(betas_test,(1,10))
        
        
        #betas = torch.Tensor(betas).to(comp_device)
        
        #body = bm(pose_body=pose_body, betas=betas_test)#, root_orient=root_orient, trans=trans)
        #AJ_test = np.asarray(c2c(body.Jtr[0]))
        #AJMov[i+1]['I'] = AJ
        #AV[i+1]['I'] = np.asarray(c2c(body.v[0]))
        if i == 0:
            #history_b = torch.cat((history_b[None,:], pose_body[None,:]), dim=1)#, axis = 1)
            #history_b_test = torch.cat((history_b_test[None,:], betas_test[None,:]), dim=1)#, axis = 1)    
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
        else:
            #history_b = torch.cat((history_b, pose_body[None,:]), dim=1)
            #history_b_test = torch.cat((history_b_test, betas_test[None,:]), dim=1)#, axis = 1)
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
            
        #print(i)
        #print(f(AJ,KJ))
        #print(history_b)
        
    return pose_body, history_b, history_f, g, stepStore, g_norm


def SteepestDescent_3(betas, pose_body, xyzKinect, N, bm, Indices):
    
    #initialize storage variables
    stepStore = []
    g_norm = []
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    
    #history_b = []
    history_f = []
  
    
    history_b = torch.cat([betas[0][None,:]], dim=1)#, axis = 1)
    #history_b_test = torch.cat([betas_i[0][None,:]], dim=1)
    history_f.append(f_3(xyzSMPL,xyzKinect,Indices))
    #history_f_test.append(f(AJ,KJ))
    
    
    
    for i in range(N):
        print("Steepest Descent Iteration")
        print(i)
        
        g = gradient_3(betas,pose_body,xyzKinect,bm,Indices) #returns a 73,1 vector
        #g_test = gradient_test(betas_test,KJ)
        g_norm.append(LA.norm(g[:,0]))
        betas = np.reshape(betas,(1,10))
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL = c2c(body.v[0])
        

        curval = f_3(xyzSMPL,xyzKinect,Indices)
        B = 2/(1 + np.sqrt(5)) #from SARA FRIDOVICH-KEIL AND BENJAMIN RECHT paper
        betas = np.reshape(betas,(10,1)) #vector
        T = 1000
        [step, fval] = findstep3(T, B, betas, pose_body, g, curval, bm, xyzKinect, Indices)
       
        #stepStore.append(step)
        #stepStore_test.append(step_test)
        
        #pose_body = pose_body - step*g
        betas = betas - step*g
 
        
        #betas_test = np.reshape(betas_test,(1,10))
        
        
        #betas = torch.Tensor(betas).to(comp_device)
        
        
        #body = bm(pose_body=pose_body, betas=betas_test)#, root_orient=root_orient, trans=trans)
        #AJ_test = np.asarray(c2c(body.Jtr[0]))
        #AJMov[i+1]['I'] = AJ
        #AV[i+1]['I'] = np.asarray(c2c(body.v[0]))
        if i == 0:
            #history_b = torch.cat((history_b[None,:], betas[None,:]), dim=1)#, axis = 1)
            #history_b_test = torch.cat((history_b_test[None,:], betas_test[None,:]), dim=1)#, axis = 1)
            
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
        else:
            #history_b = torch.cat((history_b, betas[None,:]), dim=1)
            #history_b_test = torch.cat((history_b_test, betas_test[None,:]), dim=1)#, axis = 1)
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
            
        #print(i)
        #print(f(AJ,KJ))
        #print(history_b)
        
    return betas, history_b, history_f, g, stepStore, g_norm



def SteepestDescent_4(betas, pose_body, xyzKinect, N, bm, Indices):
    
    #initialize storage variables
    #stepStore = []
    g_norm = []
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    
    #history_b = []
    history_f = []
  
    
    history_b = torch.cat([pose_body[0][None,:]], dim=1)#, axis = 1)
    #history_b_test = torch.cat([betas_i[0][None,:]], dim=1)
    history_f.append(f_3(xyzSMPL,xyzKinect, Indices))
    #history_f_test.append(f(AJ,KJ))
        
    
    for i in range(N):
        print("Steepest Descent Iteration")
        print(i)
        
        g = gradient_4(betas,pose_body,xyzKinect,bm, Indices) #returns a 73,1 vector
        #g_test = gradient_test(betas_test,KJ)
        g_norm.append(LA.norm(g[:,0]))
        
        pose_body = np.reshape(pose_body,(1,63))
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL = c2c(body.v[0])
        

        curval = f_3(xyzSMPL,xyzKinect,Indices)
        B = 2/(1 + np.sqrt(5)) #from SARA FRIDOVICH-KEIL AND BENJAMIN RECHT paper
        pose_body = np.reshape(pose_body,(63,1)) #vector
        T = 1
        [step, fval] = findstep4(T, B, betas, pose_body, g, curval, bm, xyzKinect, Indices)

        #print(np.transpose(gtemp)*gtemp)
       
        
        #alpha_1, alpha_2 =  GoldenSection_b(betas,pose_body,xyzKinect,g,bm, Sbool)
 
        #step = sum([alpha_1,alpha_2])/len([alpha_1,alpha_2])
       
        #stepStore.append(step)
        #stepStore_test.append(step_test)        
        #pose_body = pose_body - step*g
        pose_body = pose_body - step*g
 
        
        pose_body = np.reshape(pose_body,(1,63)) #reshaped to go into smpl scripts
        #betas_test = np.reshape(betas_test,(1,10))
        
        
        #betas = torch.Tensor(betas).to(comp_device)
        
        
        #body = bm(pose_body=pose_body, betas=betas_test)#, root_orient=root_orient, trans=trans)
        #AJ_test = np.asarray(c2c(body.Jtr[0]))
        #AJMov[i+1]['I'] = AJ
        #AV[i+1]['I'] = np.asarray(c2c(body.v[0]))
        if i == 0:
            #history_b = torch.cat((history_b[None,:], pose_body[None,:]), dim=1)#, axis = 1)
            #history_b_test = torch.cat((history_b_test[None,:], betas_test[None,:]), dim=1)#, axis = 1)
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
        else:
            #history_b = torch.cat((history_b, pose_body[None,:]), dim=1)
            #history_b_test = torch.cat((history_b_test, betas_test[None,:]), dim=1)#, axis = 1)
            history_f.append(fval)
            #temp = f(AJ_test,KJ)
            #history_f_test.append(temp)
            
        #print(i)
        #print(f(AJ,KJ))
        #print(history_b)
        
    return pose_body, history_b, history_f, g, g_norm