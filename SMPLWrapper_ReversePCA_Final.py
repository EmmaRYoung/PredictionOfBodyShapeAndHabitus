# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:12:30 2023

@author: swoosh
"""
import numpy as np
import torch
comp_device = torch.device("cpu")
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import json
import open3d as o3d
import time

from simpleicp import PointCloud, SimpleICP

import copy

import os

#optimization code from another folder
import sys
sys.path.insert(0,r"C:\Users\swoosh\Desktop\amass-master\Optimization Code")
from SteepestDescent import SteepestDescent_2 as SD_Pose
from SteepestDescent import SteepestDescent_3 as SD_ShapeKNN
from SteepestDescent import SteepestDescent_4 as SD_PoseKNN

from Orient import Orient

import matlab.engine
eng = matlab.engine.start_matlab()

def AndreassenCloudMorph(target_nodes, source_nodes):
    target_nodes = matlab.double(target_nodes.tolist())
    source_nodes = matlab.double(source_nodes.tolist())
    
    source_nodes1 = np.double(eng.AndreassenCloudMorph_m(target_nodes, source_nodes, nargout=1))
    
    return source_nodes1

#from cycpd github, visualize point clouds
def draw_registration_result(source, target, transformation):
    #point_size = 1
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
    
def MeanJointFrames(data):
    
   
    frame_count = 0
    for i in range(np.size(data['frames'])):
        try:
            test = np.asarray(data['frames'][i]['bodies'][0]['joint_positions'])
            frame_count = frame_count+1    
        except:
            print("Exception")
            #continue
    
    numRow = np.shape(test)[0]
    numCol = np.shape(test)[1]
    KJ_all = np.zeros((frame_count, numRow, numCol)) #preallocate (easier)
    frame_count0 = 0
    for frame in range(np.size(data['frames'])):
        try:
            KJ1 = np.asarray(data['frames'][frame]['bodies'][0]['joint_positions'])
            KJ_all[frame_count0,:,:] = KJ1
            frame_count0 = frame_count0 + 1
        except:
            continue 
        #print(frame)
        #KJ_all[frame,:,:] = KJ1
    
    #average all frames
    KJ_mean = np.mean(KJ_all, axis=0)
    KJ_mean = KJ_mean/1000
        
    return KJ_mean

#Make sure there is a point cloud and joints file for everyone you wish to process (remove extra that aren't a pair)
#Name your point cloud file with the ending specified in the below .endswith associated with PCfile
#Name your joints file with the ending specified in the below .endswith associated with Jfile
PCfile = [filename for filename in os.listdir('.') if filename.endswith("BackAdded.txt")]
Jfile = [filename for filename in os.listdir('.') if filename.endswith("Joints.json")]

#The directory where you are putting your data
Outfile = r"C:\Users\swoosh\Desktop\amass-master\MeshMorphingTest\OutData_test"

############
############ Define subject info
############
#import altered smpl verts and (unaltered) joints
relax_tally = np.zeros((10,len(PCfile)))

start = time.time()
for count in range(len(PCfile)):
    
    #S09: range(6,7)
    #S05: range(3,4)
    
    xyzKinect = np.loadtxt(PCfile[count])
    xyzKinect = xyzKinect/1000  #important! The Kinect body is in mm, the smpl body is in m
    
    JointsFile = Jfile[count]
    
    with open(JointsFile) as f:
        data = json.load(f)
        
    KJ = MeanJointFrames(data)

    
    '''  
    for frame in range(np.size(data['frames'])):
        try: 
            KJ = np.asarray(data['frames'][frame]['bodies'][0]['joint_positions'])
        except:
            print("No joints detected :(")
        KJ = KJ/1000; #important! The Kinect body is in mm, the smpl body is in m
    
    '''
    
    subID = PCfile[count][1:3]
    MFN = PCfile[count][3]
    print("Processing Subject number" + subID)
    ####Outfile information
    '''
    JointString = '\S' + subID + 'SMPLJoints.txt'
    VertString = '\S' + subID + 'SMPLVertices.txt'
    VertString1 = '\S' + subID + 'SMPLVertices_0pose.txt'
    PoseString = '\S' + subID + 'PoseParams.txt'
    BetaString = '\S' + subID + 'ShapeParams.txt'
    KPC_string = '\S' + subID + 'xyzKinect.txt'
    '''
    ############
    ############ Setup Boolean information about smpl mesh
    ############
    #Read in boolean for all genders
    NeutralBool = np.loadtxt("NeutralPCABool.txt")
    MaleBool = np.loadtxt("MalePCABool.txt")
    FemaleBool = np.loadtxt("FemalePCABool.txt")
    
    #decide what smpl model to use
    if MFN == "M":
        gender = "male"
        Bool = MaleBool
    elif MFN == "F":
        gender = "female"
        Bool = FemaleBool
    elif MFN == "N":
        gender = "neutral"
        Bool = NeutralBool
    
    #convert to a boolean
    TF = Bool == 1
    numVert = sum(TF)
    
    #test using torso boolean
    TorsoBool = np.loadtxt("TorsoBool.txt")
    TF_torso = TorsoBool == 1
    
    #feet bool
    ExtrBool = np.loadtxt("Hands_FeetBool.txt")
    TF_extr = ExtrBool == 1
    
    
    ############
    ############ Set up template instance
    ############
    bm_path = r'C:\Users\swoosh\Desktop\amass-master\support_data\body_models\smplh\{}\model.npz'.format(gender)
    dmpl_path = r'C:\Users\swoosh\Desktop\amass-master\support_data\body_models\dmpls\{}\model.npz'.format(gender)
    num_betas = 10
    num_dmpls = 8
    
    root_orient = np.zeros((1,3))
    root_orient = torch.Tensor(root_orient).to(comp_device) # controls the global root orientation
    
    trans = np.zeros((1,3))
    trans = torch.Tensor(trans).to(comp_device) #moves smpl body around in space
    
    pose_body = np.zeros((1,63))
    pose_body = torch.Tensor(pose_body).to(comp_device) # controls the body
    
    betas = np.zeros((1,10))
    betas = torch.Tensor(betas).to(comp_device) # controls the body shape
    
    bm = BodyModel(bm_fname=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_path).to(comp_device)
    body_template = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    
    joints = c2c(body_template.Jtr[0])
    AJ = np.asarray(joints)
    
        

    
    KJ, xyzKinect = Orient(AJ,KJ,xyzKinect)
  

    Template_vertices = c2c(body_template.v[0])
    #joints = c2c(body_template.Jtr[0])
    #AJ = np.asarray(joints)
    
    ############
    ############Remove vertices from template mesh, then perform ICP (rigidly transform Kinect subject point cloud)
    ############
    #remove vertices from template mesh
    TV = np.delete(Template_vertices, ~TF, axis=0) #remove rows
    #np.savetxt("TemplateMaleVertices.txt", TV, fmt = '%f')
    #np.savetxt("DefaultMaleJoints.txt", AJ, fmt = '%f')
    #np.savetxt("KinectJoints.txt", KJ, fmt = '%f')
    
    
    pc_fix = PointCloud(TV, columns=["x", "y", "z"])
    pc_mov = PointCloud(xyzKinect, columns=["x", "y", "z"])
    
    icp = SimpleICP()
    icp.add_point_clouds(pc_fix, pc_mov)
    H, xyzKinect, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=5)
    
    #transform joints also
    #add column of ones
    KJ_ = np.hstack((KJ,np.ones((len(KJ),1))))
    
    KJ_ = np.matmul(H, np.transpose(KJ_))
    KJ = KJ[:,0:3]
    
    #np.savetxt("S23_xyzKinect_icpMOD.txt", xyzKinect, fmt = '%f')
    #np.savetxt("S23_KJ_icpMOD.txt", KJ, fmt = '%f')
    #np.savetxt("FemaleVertices_MOD.txt", TV, fmt = "%f")

    
    
    
    #KPC_string = 'OutData_test\S' + subID + 'xyzKinect.txt'
    #np.savetxt(KPC_string, xyzKinect, fmt = '%f')
    #save these for figures
    #np.savetxt("PC_Oriented.txt", xyzKinect, fmt = '%f')
    

    #np.savetxt("SMPL_VertsRemoved.txt", TV, fmt = '%f')
    
    #convert back to pointcloud for visualization
    test1 = o3d.geometry.PointCloud()
    test1.points = o3d.utility.Vector3dVector(Template_vertices)
    
    xyzKinect_TPC = o3d.geometry.PointCloud()
    xyzKinect_TPC.points = o3d.utility.Vector3dVector(xyzKinect)
    
    #SMPL mesh deformed out to PC data
    trans_init = np.eye(4)
    #draw_registration_result(test1, xyzKinect_TPC, trans_init)
    
        
    #call thor's cloud morph
    TV_Trans = AndreassenCloudMorph(xyzKinect, TV)
    
    #save for ors figure?

    #np.savetxt("SMPL_VertsMorphed.txt", TV_CPD, fmt = '%f')
    
    #mean center point cloud before reverse PCA process
    TV_Transc = np.empty(TV_Trans.shape, dtype = TV_Trans.dtype)
    for i in range(3):
        TV_Transc[:,i] = TV_Trans[:,i] - np.mean(TV_Trans[:,i])
    
    
  
    #convert back to pointcloud for visualization
    test1 = o3d.geometry.PointCloud()
    test1.points = o3d.utility.Vector3dVector(TV_Trans)
    
    xyzKinect_TPC = o3d.geometry.PointCloud()
    xyzKinect_TPC.points = o3d.utility.Vector3dVector(xyzKinect)
    
    #SMPL mesh deformed out to PC data
    trans_init = np.eye(4)
    #draw_registration_result(test1, xyzKinect_TPC, trans_init)
    #np.savetxt("xyzSMPL_CPD_ORS.txt", test1.points)
    #np.savetxt("xyzKinect_CPD_ORS.txt", xyzKinect_TPC.points)

    #vectorize eigen vector matrix. Transform from 3d array into 2d
    smpl_dict = np.load(bm_path, encoding='latin1')
    shapedirs = smpl_dict['shapedirs'][:,:,:num_betas]
    
    #remove indices not used in sparce PCA - MIGHT BE WRONG????
    shapedirs_trunkate = np.delete(shapedirs, ~TF, axis=0)
    
    EVectSparse = np.empty((numVert*3, 10 ,1))
    for i in range(10):
        vector = shapedirs_trunkate[:,:,i].reshape(numVert*3,1)
        EVectSparse[:,i] = vector
    
    #calculate pseudo inverse of the eVect matrix
    EVectSparse_inv = np.linalg.pinv(EVectSparse[:,:,0])
    #print(EVectSparse[:,:,0].shape)
    
    #reshape new instance (Nx3) to a vector (3*Nx1)
    instance_reshape = TV_Transc.reshape(numVert*3,1)
    template_reshape = TV.reshape(numVert*3,1)
    #print(TV_CPDc)
    
    #calculate shape parameters in reverse PCA process
    betas = np.matmul(EVectSparse_inv,(instance_reshape - template_reshape))
    betas = np.reshape(betas,(1,10))
    
    #store which shape parameters are reset
    reset = np.zeros((1,10))
    
    #change huge outliers back to reasonable number
    
    
    for i in range(10):
        if betas[0,i] > 3:
            betas[0,i] = 3
            reset[0,i] = 1
        if betas[0,i] < -3:
            betas[0,i] = -3
            reset[0,i] = 1
    
    
    betas = torch.Tensor(betas).to(comp_device)
    
    body = bm(pose_body=pose_body, betas=betas)
    faces = c2c(bm.f)
    vertices=c2c(body.v[0])
    
    
    #convert back to pointcloud for visualization
    SMPLVerts = o3d.geometry.PointCloud()
    SMPLVerts.points = o3d.utility.Vector3dVector(vertices)
    #np.savetxt("xyzSMPL_PostRBF.txt", SMPLVerts.points)


    xyzKinect_TPC = o3d.geometry.PointCloud()
    xyzKinect_TPC.points = o3d.utility.Vector3dVector(xyzKinect)

    #SMPL mesh deformed out to PC data

    trans_init = np.eye(4)
    #draw_registration_result(SMPLVerts, xyzKinect_TPC, trans_init)
    
    
    
    #np.savetxt("TestBetas.txt", betas, fmt = "%f")
    
    #initialize variables for steepest descent 
    #solution = np.zeros((1,73))
    #solution = torch.Tensor(solution).to(comp_device)
    #solution[:,0:10] = betas
    
    #solnDiff = np.zeros((73,1))
    #solnDiff = torch.Tensor(solnDiff).to(comp_device)
    
    #indexFull = np.zeros((2,73))
    #indexFull[0, 0:10] = 1
    #indexFull[1, 10:73] = 1
    
    #save joints before optimization is done (for figures)
    #np.savetxt("SMPLJoints_b4opt.txt", np.asarray(c2c(body.Jtr[0])), fmt = "%f")
    #np.savetxt("KJ_compareopt.txt", KJ, fmt = "%f")
    

    N = 3
    pose_body, history_b, history_f, g, stepStore, g_norm = SD_Pose(betas, pose_body, KJ, N, bm)
  
    
    betas = np.reshape(betas, (1,10))
    pose_body = np.reshape(pose_body, (1,63))
    
    #body = bm(pose_body=pose_body, betas=betas)
    #np.savetxt("SMPLJoints_afteropt.txt", np.asarray(c2c(body.Jtr[0])), fmt = "%f")
    
    #save joints after optimization is done (for figures)
    
    #print(np.shape(pose_body))
    #print(np.shape(betas))
    #pose_body = np.reshape(pose_body, (1,63))
    
    hist = np.zeros((1,1))
    
    cycles = 75
    #cycles = 25
    for i in range(cycles):
        print("cycle number:")
        print(i)
        N1 = 1
        betas, history_b, history_f, g, stepStore, g_norm = SD_ShapeKNN(betas, pose_body, xyzKinect, N1, bm, TF, TF_torso, TF_extr)
    
        betas = np.reshape(betas, (1,10))
        pose_body = np.reshape(pose_body, (1,63))
        history_f = history_f[0:-1]
        hist = np.vstack((hist, history_f))
        
    
        N2 = 1
        pose_body, history_b, history_f0, g, g_norm = SD_PoseKNN(betas, pose_body, xyzKinect, N2, bm, TF, TF_torso, TF_extr)
        history_f0 = history_f0[0:-1]
        hist = np.vstack((hist, history_f0))
       
        
        for j in range(10):
            if betas[0,j] > 3:
                betas[0,j] = 1.5
                relax_tally[j,count] = relax_tally[j,count] + 1
            if betas[0,j] < -3:
                betas[0,j] = -1.5
                relax_tally[j,count] = relax_tally[j,count] + 1
        
        
        
    #graph objective function
    #hist_ = np.delete(hist, 0)
    #import matplotlib.pyplot as plt 
    #fig, ax = plt.subplots() 
    #ax.plot(np.arange(len(hist_)), hist_,'r')
    #ax.plot(np.arange(20), hist_[0:20],'r')
    
    #save info
    betas = np.reshape(betas, (1,10))
    pose_body = np.reshape(pose_body, (1,63))
    body = bm(pose_body=pose_body, betas=betas)
    
    VertString = 'OutData_test/S' + subID + 'SMPLVertices_release.txt'
    BetaString = 'OutData_test/S' + subID + 'ShapeParams_release.txt'
    KPC_string = 'OutData_test/S' + subID + 'xyzKinect.txt'

    '''
    np.savetxt(VertString, c2c(body.v[0]), fmt = '%f')
    np.savetxt(BetaString, betas, fmt = '%f')
    np.savetxt(KPC_string, xyzKinect, fmt = '%f')
    '''
    
    body = bm(pose_body=pose_body, betas=betas)#, root_orient=root_orient, trans=trans)
    xyzSMPL= c2c(body.v[0])

    
    #convert back to pointcloud for visualization
    SMPLVerts = o3d.geometry.PointCloud()
    SMPLVerts.points = o3d.utility.Vector3dVector(xyzSMPL)

    xyzKinect_TPC = o3d.geometry.PointCloud()
    xyzKinect_TPC.points = o3d.utility.Vector3dVector(xyzKinect)

    #SMPL mesh deformed out to PC data

    trans_init = np.eye(4)
    #draw_registration_result(SMPLVerts, xyzKinect_TPC, trans_init)
    
    end = time.time()
    print(end-start)
    
    

eng.quit()

