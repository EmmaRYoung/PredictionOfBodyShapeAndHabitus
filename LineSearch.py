# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:12:57 2022

@author: Emma.R.Young
"""

#APPROXIMATELY EXACT LINE SEARCH∗
#SARA FRIDOVICH-KEIL† AND BENJAMIN RECHT†
#adapted function call but all the same

#import os
#import scipy
#from scipy import optimize
#import numpy as np
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import numpy as np
#from ObjFcn import f_1
from ObjFcn import f_2
from ObjFcn import f_3

'''
def find_stepsize_approx_exact_1(T, beta, x, pose_body, direction, curval, bm, KJ):
    #x must be a column vector, is reshaped into row to go into smpl scripts
    #x is my variable beta, but beta is already used 
    #print("in step fcn")
    t = T
    t_old = 0
    f_old_old = curval
    f_old = curval
    
    #xnew
    xnew = x - t*direction
    #reshape to go into smpl scripts
    xnew = np.reshape(xnew,(1,10)) 
    body = bm(pose_body=pose_body, betas=xnew)#, root_orient=root_orient, trans=trans)
    
    #function eval
    AJ = np.asarray(c2c(body.Jtr[0]))
    f_new = f_1(AJ,KJ)
    
    #f_new = f.val(x - t * direction, batch=batch)
    
    f_first = f_new
    
    if f_new is None:
        return 0, f_old
    
    param = beta
    
    if f_new <= f_old:
        param = 1.0 / beta
        
    num_iters = 0
    count = 0
    
    while True:
        num_iters = num_iters + 1
        t_old_old = t_old
        t_old = t
        t = t * param
        f_old_old = f_old
        f_old = f_new
        
        xnew = x - t*direction
        xnew = np.reshape(xnew,(1,10)) 
        body = bm(pose_body=pose_body, betas=xnew)#, root_orient=root_orient, trans=trans)
        
        #function eval
        AJ = np.asarray(c2c(body.Jtr[0]))
        f_new = f_1(AJ,KJ)
            
        if f_new is None:
            return 0, curval
        if f_new > curval and param < 1: # Special case for nonconvex functions to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing step size
            count = count + 1
        if count > 20: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
        
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t = t_old # Ok to stop updating t_old, t_old_old, and f_old_old once we're backtracking
        param = beta
        f_old, f_new = f_new, f_old
        
        if count > 20: # Tried increasing step size, but function was flat
            t = T
            f_new = f_first
        count = 0
        
        while True:
            t = t * param
            f_old = f_new
            
            xnew = x - t*direction
            xnew = np.reshape(xnew,(1,10)) 
            body = bm(pose_body=pose_body, betas=xnew)#, root_orient=root_orient, trans=trans)
            
            #function eval
            AJ = np.asarray(c2c(body.Jtr[0]))
            f_new = f_1(AJ,KJ)
            
            if f_new is None:
                return 0, curval
            if f_new == f_old:
                count = count + 1
            if count > 20: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old:
                break
    if param < 1:
        return t, f_new
    return t_old_old, f_old_old
'''
def find_stepsize_approx_exact_2(T, beta, betas, x, direction, curval, bm, KJ):
    #x must be a column vector, is reshaped into row to go into smpl scripts
    #x is my variable beta, but beta is already used 
    #print("in step fcn")
    t = T
    t_old = 0
    f_old_old = curval
    f_old = curval
    
    #xnew
    xnew = x - t*direction
    #reshape to go into smpl scripts
    xnew = np.reshape(xnew,(1,63)) 
    body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
    
    #function eval
    AJ = np.asarray(c2c(body.Jtr[0]))
    f_new = f_2(AJ,KJ)
    
    #f_new = f.val(x - t * direction, batch=batch)
    
    f_first = f_new
    
    if f_new is None:
        return 0, f_old
    
    param = beta
    
    if f_new <= f_old:
        param = 1.0 / beta
        
    num_iters = 0
    count = 0
    
    while True:
        num_iters = num_iters + 1
        t_old_old = t_old
        t_old = t
        t = t * param
        f_old_old = f_old
        f_old = f_new
        
        xnew = x - t*direction
        xnew = np.reshape(xnew,(1,63)) 
        body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
        
        #function eval
        AJ = np.asarray(c2c(body.Jtr[0]))
        f_new = f_2(AJ,KJ)
            
        if f_new is None:
            return 0, curval
        if f_new > curval and param < 1: # Special case for nonconvex functions to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing step size
            count = count + 1
        if count > 20: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
        
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t = t_old # Ok to stop updating t_old, t_old_old, and f_old_old once we're backtracking
        param = beta
        f_old, f_new = f_new, f_old
        
        if count > 20: # Tried increasing step size, but function was flat
            t = T
            f_new = f_first
        count = 0
        
        while True:
            t = t * param
            f_old = f_new
            
            xnew = x - t*direction
            xnew = np.reshape(xnew,(1,63)) 
            body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
            
            #function eval
            AJ = np.asarray(c2c(body.Jtr[0]))
            f_new = f_2(AJ,KJ)
            
            if f_new is None:
                return 0, curval
            if f_new == f_old:
                count = count + 1
            if count > 20: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old:
                break
    if param < 1:
        return t, f_new
    return t_old_old, f_old_old

def find_stepsize_approx_exact_3(T, beta, x, pose_body, direction, curval, bm, xyzKinect, Indices, TorsoBool, ExtrBool):
    #x must be a column vector, is reshaped into row to go into smpl scripts
    #x is my variable beta, but beta is already used 
    #print("in step fcn")
    t = T
    t_old = 0
    f_old_old = curval
    f_old = curval
    
    #xnew
    xnew = x - t*direction
    #reshape to go into smpl scripts
    xnew = np.reshape(xnew,(1,10)) 
    body = bm(pose_body=pose_body, betas=xnew)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    #function eval
    f_new = f_3(xyzSMPL, xyzKinect, xnew, pose_body, Indices, TorsoBool, ExtrBool)
    
    #f_new = f.val(x - t * direction, batch=batch)
    
    f_first = f_new
    
    if f_new is None:
        return 0, f_old
    
    param = beta
    
    if f_new <= f_old:
        param = 1.0 / beta
        
    num_iters = 0
    count = 0
    
    while True:
        num_iters = num_iters + 1
        t_old_old = t_old
        t_old = t
        t = t * param
        f_old_old = f_old
        f_old = f_new
        
        xnew = x - t*direction
        xnew = np.reshape(xnew,(1,10)) 
        body = bm(pose_body=pose_body, betas=xnew)#, root_orient=root_orient, trans=trans)
        xyzSMPL = c2c(body.v[0])
        #function eval
        f_new = f_3(xyzSMPL, xyzKinect, xnew, pose_body, Indices, TorsoBool, ExtrBool)
            
        if f_new is None:
            return 0, curval
        if f_new > curval and param < 1: # Special case for nonconvex functions to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing step size
            count = count + 1
        if count > 20: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
        
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t = t_old # Ok to stop updating t_old, t_old_old, and f_old_old once we're backtracking
        param = beta
        f_old, f_new = f_new, f_old
        
        if count > 20: # Tried increasing step size, but function was flat
            t = T
            f_new = f_first
        count = 0
        
        while True:
            t = t * param
            f_old = f_new
            
            xnew = x - t*direction
            xnew = np.reshape(xnew,(1,10)) 
            body = bm(pose_body=pose_body, betas=xnew)#, root_orient=root_orient, trans=trans)
            xyzSMPL = c2c(body.v[0])
            #function eval
            f_new = f_3(xyzSMPL, xyzKinect, xnew, pose_body, Indices, TorsoBool, ExtrBool)
            
            if f_new is None:
                return 0, curval
            if f_new == f_old:
                count = count + 1
            if count > 20: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old:
                break
    if param < 1:
        return t, f_new
    return t_old_old, f_old_old

def find_stepsize_approx_exact_4(T, beta, betas, x, direction, curval, bm, xyzKinect, Indices, TorsoBool, ExtrBool):
    #x must be a column vector, is reshaped into row to go into smpl scripts
    #x is my variable beta, but beta is already used 
    #print("in step fcn")
    t = T
    t_old = 0
    f_old_old = curval
    f_old = curval
    
    #xnew
    xnew = x - t*direction
    #reshape to go into smpl scripts
    xnew = np.reshape(xnew,(1,63)) 
    body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    #function eval
    f_new = f_3(xyzSMPL, xyzKinect, betas, xnew, Indices, TorsoBool, ExtrBool)
    
    #f_new = f.val(x - t * direction, batch=batch)
    
    f_first = f_new
    
    if f_new is None:
        return 0, f_old
    
    param = beta
    
    if f_new <= f_old:
        param = 1.0 / beta
        
    num_iters = 0
    count = 0
    
    while True:
        num_iters = num_iters + 1
        t_old_old = t_old
        t_old = t
        t = t * param
        f_old_old = f_old
        f_old = f_new
        
        xnew = x - t*direction
        xnew = np.reshape(xnew,(1,63)) 
        body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL = c2c(body.v[0])
        #function eval
        f_new = f_3(xyzSMPL, xyzKinect, betas, xnew, Indices, TorsoBool, ExtrBool)
        if f_new is None:
            return 0, curval
        if f_new > curval and param < 1: # Special case for nonconvex functions to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing step size
            count = count + 1
        if count > 20: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
        
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t = t_old # Ok to stop updating t_old, t_old_old, and f_old_old once we're backtracking
        param = beta
        f_old, f_new = f_new, f_old
        
        if count > 20: # Tried increasing step size, but function was flat
            t = T
            f_new = f_first
        count = 0
        
        while True:
            t = t * param
            f_old = f_new
            
            xnew = x - t*direction
            xnew = np.reshape(xnew,(1,63)) 
            body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
            xyzSMPL = c2c(body.v[0])
            #function eval
            f_new = f_3(xyzSMPL, xyzKinect, betas, xnew, Indices, TorsoBool, ExtrBool)
            
            if f_new is None:
                return 0, curval
            if f_new == f_old:
                count = count + 1
            if count > 20: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old:
                break
    if param < 1:
        return t, f_new
    return t_old_old, f_old_old

'''
def find_stepsize_approx_exact_all(T, beta, x, direction, curval, bm, xyzKinect, Indices):
    #x must be a column vector, is reshaped into row to go into smpl scripts
    #x is my variable beta, but beta is already used 
    #print("in step fcn")
    x = np.reshape(x, (73,1))
    
    t = T
    t_old = 0
    f_old_old = curval
    f_old = curval
    
    #xnew
    xnew = x - t*direction
    #reshape to go into smpl scripts
    xnew = np.reshape(xnew,(1,73)) 
    betas = xnew[:,0:10]
    pose_body = xnew[:,10:73]
    
    body = bm(pose_body=pose_body, betas=betas)
    xyzSMPL = c2c(body.v[0])
    #function eval
    f_new = f_3(xyzSMPL,xyzKinect, Indices)
    
    #f_new = f.val(x - t * direction, batch=batch)
    
    f_first = f_new
    
    if f_new is None:
        return 0, f_old
    
    param = beta
    
    if f_new <= f_old:
        param = 1.0 / beta
        
    num_iters = 0
    count = 0
    
    xnew = np.reshape(xnew,(73,1))
    while True:
        num_iters = num_iters + 1
        t_old_old = t_old
        t_old = t
        t = t * param
        f_old_old = f_old
        f_old = f_new
        
        xnew = x - t*direction
        xnew = np.reshape(xnew,(1,73)) 
  
        
        betas = xnew[:,0:10]
        pose_body = xnew[:,10:73]

        
        body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
        xyzSMPL = c2c(body.v[0])
        #function eval
        f_new = f_3(xyzSMPL,xyzKinect,Indices)
        if f_new is None:
            return 0, curval
        if f_new > curval and param < 1: # Special case for nonconvex functions to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing step size
            count = count + 1
        if count > 20: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
        
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t = t_old # Ok to stop updating t_old, t_old_old, and f_old_old once we're backtracking
        param = beta
        f_old, f_new = f_new, f_old
        
        if count > 20: # Tried increasing step size, but function was flat
            t = T
            f_new = f_first
        count = 0
        
        while True:
            t = t * param
            f_old = f_new
            
            xnew = x - t*direction
            xnew = np.reshape(xnew,(1,73)) 
            betas = xnew[:,0:10]
            pose_body = xnew[:,10:73]
            
            body = bm(pose_body=xnew, betas=betas)#, root_orient=root_orient, trans=trans)
            xyzSMPL = c2c(body.v[0])
            #function eval
            f_new = f_3(xyzSMPL,xyzKinect,Indices)
            
            if f_new is None:
                return 0, curval
            if f_new == f_old:
                count = count + 1
            if count > 20: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old:
                break
    if param < 1:
        return t, f_new
    return t_old_old, f_old_old
'''