# -*- coding: utf-8 -*-
#!/usr/bin/env python
# SMPL SHAPE WRAPER


#import

import numpy as np
import torch
comp_device = torch.device("cpu")
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import json
import time
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SteepestDescent import SteepestDescent_1
from SteepestDescent import SteepestDescent_2
from SteepestDescent import SteepestDescent_3
from SteepestDescent import SteepestDescent_4

from Orient import Orient


#create filename strings
subID = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']

subGender = ['male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'female', 'male', 'female', 'male']    


#remove subjects where there are current problems with point clouds


Indices = np.loadtxt('FrontRighLeft_Indices.txt', delimiter = ',', dtype = 'int')
Indices = Indices - 1



for i in range(len(subID)):
    
    #Out file info - path
    JointString = 'SubjectPC/OutData1/S' + subID[i] + 'SMPLJoints.txt'
    VertString = 'SubjectPC/OutData1/S' + subID[i] + 'SMPLVertices.txt'
    KPC_path = 'SubjectPC/OutData1/S' + subID[i] + 'xyzKinect.txt'
    beta_path = 'SubjectPC/OutData1/S' + subID[i] + 'Betas.txt'

    #subject info - path
    SubjectPC = 'SubjectPC/S' + subID[i] + 'CompleteTpose_Mirrored.txt'
    JointsFile = 'SubjectPC/S' + subID[i] + 'TposeJoints.json'
    
    
    print("Processing Subject: ", subID[i])
        
    
    ## setup path to body model
    bm_path = '../support_data/body_models/smplh/' + subGender[i] + '/model.npz'
    dmpl_path = '../support_data/body_models/dmpls/' + subGender[i] + '/model.npz'
    num_betas = 10 # number of body parameters
    num_dmpls = 8
    
    
    #initialize SMPL instance
    root_orient = np.zeros((1,3))
    root_orient = torch.Tensor(root_orient).to(comp_device) # controls the global root orientation
    
    trans = np.zeros((1,3))
    trans = torch.Tensor(trans).to(comp_device) #moves smpl body around in space
    
    pose_body = np.zeros((1,63))
    pose_body = torch.Tensor(pose_body).to(comp_device) # controls the body
    
    betas_i = np.zeros((1,10))
    betas_i = torch.Tensor(betas_i).to(comp_device) # controls the body shape
    
    bm = BodyModel(bm_fname=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_path).to(comp_device)
    
    body = bm(pose_body=pose_body, betas=betas_i)#, root_orient=root_orient, trans=trans)
    
    joints = c2c(body.Jtr[0])
    vertices = c2c(body.v[0])
    faces = c2c(bm.f)
    
    
    AJ = np.asarray(joints)
    AF = np.asarray(faces) #do only once, these never change
    
    #Retreive Subject's Kinect information
    try:
        with open(JointsFile) as f:
            data = json.load(f)
    except:
        continue
    
    #Kinect Joints
    KJ = np.asarray(data['frames'][8]['bodies'][0]['joint_positions'])
    KJ = KJ/1000; #important! The Kinect body is in mm, the smpl body is in m
    
    #Read in subject point cloud
    xyzKinect = np.loadtxt(SubjectPC, delimiter = ',')
    xyzKinect = xyzKinect/1000
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(KJ[:,0], KJ[:,1], KJ[:,2])
    ax.scatter(xyzKinect[:,0], xyzKinect[:,1], xyzKinect[:,2])
    ax.view_init(-90,0)
    ax.set_title('1')
    plt.show()
    
    
    #transform mesh and point cloud into the same coordinate system
    KJ, xyzKinect = Orient(AJ,KJ,xyzKinect)
    np.savetxt(KPC_path, xyzKinect, fmt='%f')
    
    
    ## Start Steepest Descent sequence
    N = 50
    
    start = time.time()
    Sbool = 1
    
    start1 = time.time()
    #Adjusts the limb lengths of the SMPL body
    betas, history_b, history_f1, g, stepStore, g_norm = SteepestDescent_1(betas_i, pose_body, KJ, N, bm)
    end = time.time()
    print("elapsed time1")
    print(end- start1)
    
    betas = np.reshape(betas,(1,10))
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    AJ = np.asarray(c2c(body.Jtr[0]))
    
    
    
    pose_body = np.zeros((1,63))
    pose_body = torch.Tensor(pose_body).to(comp_device) # controls the body
    
    N = 50
    start = time.time()
    #Roughly adjust pose  of SMPL body with the skeletal joints
    pose_body, history_b, history_f2, g1, stepStore1, g_norm1 = SteepestDescent_2(betas, pose_body, KJ, N, bm)
    end = time.time()
    print("elapsed time2")
    print(end - start)
    
    pose_body = np.reshape(pose_body,(1,63))
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    AJ = np.asarray(c2c(body.Jtr[0]))
        
    ############################################
    #The objective functions now judge the distance between the SMPL body surface and the Kinect point cloud with a nearest neighbor algorithm
    ############################################3
    
    N = 10
    start = time.time()
    #Quickly adjust shape of SMPL model
    betas, history_b, history_f3, g, stepStore, g_norm = SteepestDescent_3(betas, pose_body, xyzKinect, N, bm, Indices)
    end = time.time()
    print("elapsed time3")
    print(end- start)
    
    betas = np.reshape(betas,(1,10))
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])

    #xyzKinect = np.loadtxt('KinectVerts_Arms.txt')
    
    N = 5
    start = time.time()
    #Quickly adjust pose of the SMPL model
    pose_body, history_b, history_f4, g, g_norm = SteepestDescent_4(betas, pose_body, xyzKinect, N, bm, Indices)
    end = time.time()
    print("elapsed time4")
    print(end- start)
    
    pose_body = np.reshape(pose_body,(1,63))
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    
    
    
    N = 100
    start = time.time()
    #Perform many steepest descent iterations to "shrink wrap" the SMPL person to the Kinect point cloud
    betas, history_b, history_f, g, stepStore, g_norm = SteepestDescent_3(betas, pose_body, xyzKinect, N, bm, Indices)
    end = time.time()
    print("elapsed time5")
    print(end- start)
    
    betas = np.reshape(betas,(1,10))
    
    body = bm(pose_body=Tpose, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL = c2c(body.v[0])
    AJ = np.asarray(c2c(body.Jtr[0]))
    np.savetxt(VertString, c2c(body.v[0]), fmt = '%f')
    np.savetxt(JointString, AJ, fmt = '%f')
    np.savetxt(beta_path, betas, fmt = '%f')
    
    
    
    
    
    print("Total Time")
    print(end - start1)
    
    '''
    itercount = len(history_f1)
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(itercount), history_f1,'b')
    ax1.set(xlabel = 'Iteration')
    ax1.set_title('StepSize4')
    plt.show()
    '''


        


#plot results




'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)


#ax = fig.add_subplot(projection='3d')
ax.scatter(AJ[:,0], AJ[:,1], AJ[:,2])
ax.scatter(KJ[:,0], KJ[:,1], KJ[:,2])
ax.view_init(-90,0)

plt.show()


itercount = len(history_f)
fig, ax1 = plt.subplots()
ax1.plot(np.arange(itercount), history_f,'b')
ax1.set(xlabel = 'Iteration')
ax1.set_title('Function Value: Pose with NN')

#itercount = len(history_f)
#fig, ax1 = plt.subplots()
#ax1.plot(np.arange(itercount), history_f1,'b')
#ax1.set(xlabel = 'Iteration')
#ax1.set_title('Parameter')

#fig, ax4 = plt.subplots()
#itercount = len(stepStore)
#ax4.plot(np.arange(itercount), stepStore, 'b')
#ax4.plot(np.arange(itercount), g_norm1,'r')
#ax4.plot(np.arange(itercount), g_norm2, 'c')
#ax4.set(xlabel = 'Iteration')
#ax4.set_title('StepStore: Shape and Pose')


#fig, ax4 = plt.subplots()
#itercount = len(stepStore1)
#ax4.plot(np.arange(itercount), stepStore1, 'b')
#ax4.plot(np.arange(itercount), g_norm1,'r')
#ax4.plot(np.arange(itercount), g_norm2, 'c')
#ax4.set(xlabel = 'Iteration')
#ax4.set_title('StepStore: Pose')

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

body = bm(pose_body=pose_body, betas=betas)
body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
mv.set_static_meshes([body_mesh])
body_image = mv.render(render_wireframe=False)
show_image(body_image)
'''
'''
itercount = len(history_b[0])


fig, axs1 = plt.subplots(5,1, sharex = True)
axs1[0].plot(np.arange(itercount),(np.array(history_b[0][:,0])),'b')
axs1[0].plot(np.arange(itercount),(np.array(history_b1[0][:,0])),'r')
axs1[0].plot(np.arange(itercount),(np.array(history_b2[0][:,0])),'c')
#axs1[0].plot(np.arange(itercount),(np.array(history_b3[0][:,0])),'y')
axs1[0].set_title(r'Changes in $\beta_1$')

axs1[1].plot(np.arange(itercount),(np.array(history_b[0][:,1])),'b')
axs1[1].plot(np.arange(itercount),(np.array(history_b1[0][:,1])),'r')
axs1[1].plot(np.arange(itercount),(np.array(history_b2[0][:,1])),'c')
#axs1[1].plot(np.arange(itercount),(np.array(history_b3[0][:,1])),'y')
axs1[1].set_title(r'Changes in $\beta_2$')

axs1[2].plot(np.arange(itercount),(np.array(history_b[0][:,2])),'b')
axs1[2].plot(np.arange(itercount),(np.array(history_b1[0][:,2])),'r')
axs1[2].plot(np.arange(itercount),(np.array(history_b2[0][:,2])),'c')
#axs1[2].plot(np.arange(itercount),(np.array(history_b3[0][:,2])),'y')
axs1[2].set_title(r'Changes in $\beta_3$')

axs1[3].plot(np.arange(itercount),(np.array(history_b[0][:,3])),'b')
axs1[3].plot(np.arange(itercount),(np.array(history_b1[0][:,3])),'r')
axs1[3].plot(np.arange(itercount),(np.array(history_b2[0][:,3])),'c')
#axs1[3].plot(np.arange(itercount),(np.array(history_b3[0][:,3])),'y')
axs1[3].set_title(r'Changes in $\beta_4$')

axs1[4].plot(np.arange(itercount),(np.array(history_b[0][:,4])),'b')
axs1[4].plot(np.arange(itercount),(np.array(history_b1[0][:,4])),'r')
axs1[4].plot(np.arange(itercount),(np.array(history_b2[0][:,4])),'c')
#axs1[4].plot(np.arange(itercount),(np.array(history_b3[0][:,4])),'y')
axs1[4].set_title(r'Changes in $\beta_5$')

fig, axs2 = plt.subplots(5,1, sharex = True)
axs2[0].plot(np.arange(itercount),(np.array(history_b[0][:,5])),'b')
axs2[0].plot(np.arange(itercount),(np.array(history_b1[0][:,5])),'r')
axs2[0].plot(np.arange(itercount),(np.array(history_b2[0][:,5])),'c')
#axs2[0].plot(np.arange(itercount),(np.array(history_b3[0][:,5])),'y')
axs2[0].set_title(r'Changes in $\beta_6$')

axs2[1].plot(np.arange(itercount),(np.array(history_b[0][:,6])),'b')
axs2[1].plot(np.arange(itercount),(np.array(history_b1[0][:,6])),'r')
axs2[1].plot(np.arange(itercount),(np.array(history_b2[0][:,6])),'c')
#axs2[1].plot(np.arange(itercount),(np.array(history_b3[0][:,6])),'y')
axs2[1].set_title(r'Changes in $\beta_7$')

axs2[2].plot(np.arange(itercount),(np.array(history_b[0][:,7])),'b')
axs2[2].plot(np.arange(itercount),(np.array(history_b1[0][:,7])),'r')
axs2[2].plot(np.arange(itercount),(np.array(history_b2[0][:,7])),'c')
#axs2[2].plot(np.arange(itercount),(np.array(history_b3[0][:,7])),'y')
axs2[2].set_title(r'Changes in $\beta_8$')

axs2[3].plot(np.arange(itercount),(np.array(history_b[0][:,8])),'b')
axs2[3].plot(np.arange(itercount),(np.array(history_b1[0][:,8])),'r')
axs2[3].plot(np.arange(itercount),(np.array(history_b2[0][:,8])),'c')
#axs2[3].plot(np.arange(itercount),(np.array(history_b3[0][:,8])),'y')
axs2[3].set_title(r'Changes in $\beta_9$')

axs2[4].plot(np.arange(itercount),(np.array(history_b[0][:,9])),'b')
axs2[4].plot(np.arange(itercount),(np.array(history_b1[0][:,9])),'r')
axs2[4].plot(np.arange(itercount),(np.array(history_b2[0][:,9])),'c')
#axs2[4].plot(np.arange(itercount),(np.array(history_b3[0][:,9])),'y')
axs2[4].set_title(r'Changes in $\beta_{10}$')


for ax in axs1.flat:
    ax.set(xlabel='Iteration')
#plt.plot(np.arange(itercount),(np.array(history_b[0][:,0])),'b')

for ax in axs1.flat:
        ax.label_outer()

#fig.tight_layout()

itercount = len(history_f)
fig, ax = plt.subplots()
ax.plot(np.arange(itercount), history_f,'b')
ax.set(xlabel = 'Iteration')
ax.set_title('Function Value')
#ax.plot(np.arange(itercount), history_f1,'r')
#ax.plot(np.arange(itercount), history_f2,'c')
#ax.plot(np.arange(itercount), history_f3,'y')

itercount = len(stepStore)

fig, ax3 = plt.subplots()
ax3.plot(np.arange(itercount), stepStore)
ax3.set(xlabel = 'Iteration')
ax3.set_title('StepSize')

fig, ax4 = plt.subplots()
ax4.plot(np.arange(itercount), g_norm, 'b')
#ax4.plot(np.arange(itercount), g_norm1,'r')
#ax4.plot(np.arange(itercount), g_norm2, 'c')
ax4.set(xlabel = 'Iteration')
ax4.set_title('Norm of Gradient')
'''
