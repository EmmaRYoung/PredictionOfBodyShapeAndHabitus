# PredictionOfBodyShape_SMPL
Thesis work to develop a noninvasive way of predicting a subject's body shape using data from Azure Kinect depth cameras and the SMPL body model.

[Download a PDF of the poster here](https://tinyurl.com/bdctxjku)

[Download my thesis through ProQuest](https://www.proquest.com/docview/2840141315)

Please contact me if you are unable to access the full manuscript: emma.young.r@gmail.com OR emma.r.young@du.edu

![S02Gif](https://user-images.githubusercontent.com/67296859/218332402-98c949fc-e8ad-4844-a71b-65745d4b6e06.gif) 

# Acknowledgements 
This project was supported in part by a University of Denver Professional Research Opportunities for Faculty grant

Thank you to my colleagues for their expert advice and support:

Peter Laz, Casey Myers, Paul Rullkoetter

This project would not be possible without Abigail Eustace's thesis about the Azure Kinect camera. Thank you for your hard work!

Thank you Mom!

# Requirements
[The SMPL body model and its requirements](https://smpl.is.tue.mpg.de/)

[cycpd](https://github.com/gattia/cycpd)
* My code uses a proprietary radial basis function (RBF) method to perform mesh morphing quickly. This code is not included in this repo. Comperable results can be achieved using this coherent point drift (CPD) library.

[sklearn](https://scikit-learn.org/stable/index.html) 

# Methods
## High level overview

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/690f1c11-27bc-4756-b532-5da98f765411)


### Point Cloud and Key Point Data
1. "Scan" subject with two azure kinect cameras and physically collect anthropometric measures with a tailor's tape for validation.
   
| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/37651069-8549-4409-8dc1-9bf2df121769)
| :--:
| Camera placement for data collection

| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/6d48fa08-811f-46e5-bc21-cdb5aef4b2b0)
| :--:
| Resulting point cloud after two views are combined

Anthropometric measures
| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/2bddacd5-aa60-4761-aa12-5aa57638d465) | ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/942f966d-0095-4b94-b5c3-cc2f992f3804) | ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/ffacb928-759d-4ff2-abd5-3b4cbcef951c)
| :--: | :--: | :--:
| The circumference measurements taken of each subject | The length measurements taken of each subject | The alternative chest measurements taken of female subjects


2. Recreate missing sections of the subject point cloud (PC).
Originally, the subject PC only includes data of the front and right side. The left side is created by mirroring the right side data. The back is estimated by connecting the back most points on the left and right side. Then, the pointcloud is downsampled. Code for this data processing method is not included here.   

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/afb5db8f-88e4-4102-9929-73d4a46d09c7)

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/3ad577d7-4cf2-4031-a0ec-5da666f1c93d)

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/bccc00c0-bc33-4fe2-8d7f-9b40e85108e7)

### Rigid Alignment

1. Remove vertices from the template SMPL mesh that correspond to incomplete PC data. Removed indices are shown in red on the template male SMPL model below.
Regions of incomplete PC data are the upper back, back of arms, and back of legs. 4136 of the 6890 vertices remain.

![image](https://user-images.githubusercontent.com/67296859/218335092-cf6a6f0b-09e5-4930-8109-83e0a0b8f7b4.png) 

2. Align Kinect point cloud and joints with the SMPL mesh vertices and joints.
* Kabsch for rough alignment - Align Kinect joints with SMPL skeleton
  
| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/3a8517c0-2a40-4d0d-b6e7-0aeaa977bf78)
| :--:
| Kinect data is initially in a different coordinate system to the SMPL model


| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/d83f30ca-2c86-4739-a033-bf928e0b2fa7)
| :--:
| The transformation matrix found from the Kabsch algorithm is applied to the Kinect skeleton (left) and also the Kinect subject point cloud (right)

* ICP for fine refinement - Align Kinect point cloud with SMPL mesh vertices
  
![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/6c9848ee-0e20-4196-b58a-b3be51d4ada2)



### RBF Mesh Morphing
1. Deform the modified SMPL mesh out to the subject point cloud using a radial basis mesh morphing code. This is not provided, but comparable results can be achieved with a CPD algorithm 

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/6550671d-f4e3-4023-8531-3eeb6cdd9fe6)


2. Extract the principal component scores, or betas, associated with a new instance using the eigen vector matrix provided in the SMPL Python scripts. 

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/e675749c-4bd6-47cf-b209-4a43d6dbcd67)



### Optimization

1. Rough pose alignment between SMPL skeleton and Kinect joints

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/f03a35de-162e-44d4-a1f9-fd65d749f255)


2. K-Nearest Neighbor Optimization 

| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/e54baa14-cbc5-414d-8210-12a43cd7940c)
| :--:
| Objective Function: Minimize the distance between SMPL mesh and Kinect point cloud​

| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/814c8d23-7e4d-489b-a643-0584754195ca)
| :--:
| Varables of SMPL shape and pose are alternated in this optimization process. 

# Results
There was a positive correlation between subject manual measurements and those generated from the optimized SMPL instance fit with the partial data with an R<sup>2</sup> = 0.99 and P << 0.01 

| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/e3b4fea4-19db-4bc8-957d-b408777bb427)
| :--:
| Model-predicted and manual anthropometric​ measures compared for all measures and subjects


The validity of the model was further quantified with a percent error calculation between subject manual and SMPL measurements. A Bland-Altman chart is used to quantify the agreement between the manual and model-based anthropometric measurements across all measures and subjects. Across all measures, there is an average absolute percent difference of 4.71 ± 4.10% and the average absolute percent error between any two measures never exceeds 10% 

| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/7154ddc0-39fc-4ff8-984a-c4815a09309c)
| :--:
| Bland-Altman diagram of all model-predicted and physical anthropometric measures. 

The average absolute percent error for every measure
| . | Mean | SD
| :--: | :--: | :--:
| Floor to Shoulder | 1.17 | 0.76
| Height | 1.47 | 0.80 
| Butt | 2.74 | 2.15
| Forearm | 2.78 | 2.11
| Waist | 3.29 | 2.46
| Armpit to Wrist | 3.35 | 2.00
| Groin to Floor | 3.66 | 2.88
| Wrist | 3.71 | 3.20
| Cup Size | 4.35 | 4.81
| Band Size | 4.41 | 2.06
| Calf Upper | 5.03 | 3.28
| Bicep Upper | 5.26 | 4.09
| Thigh Upper | 5.47 | 3.65
| Chest | 5.61 | 2.35
| Bicep Lower | 5.73 | 3.93
| Calf Largest | 8.17 | 4.36
| Thigh Lower | 9.19 | 5.12
| Ankle | 9.80 | 6.33



![S11Gif](https://user-images.githubusercontent.com/67296859/218332432-0edd0de7-8b55-4e26-bb3a-1f9f71b6c103.gif) 
![S20Gif](https://user-images.githubusercontent.com/67296859/218332629-72865de4-bc88-4301-b0c6-f64767d11ad8.gif)
![S16Gif](https://user-images.githubusercontent.com/67296859/218332677-bff3f9f8-f339-4d18-8f1b-b96dddca54a6.gif)
![S05Gif 1](https://user-images.githubusercontent.com/67296859/218332913-d2bb02e3-c7d7-4d5c-b061-9c60496adec5.gif)
![S06Gif](https://user-images.githubusercontent.com/67296859/218333127-aedc46b3-5035-4a0f-8ba6-bf6774e34685.gif)
![S22Gif](https://user-images.githubusercontent.com/67296859/218332864-4648edcc-d5fa-4eaa-b31e-38d3a80d3038.gif)

