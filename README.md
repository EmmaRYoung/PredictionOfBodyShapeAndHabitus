# PredictionOfBodyShape_SMPL
Thesis work to develop a noninvasive way of predicting a subject's body shape using data from Azure Kinect depth cameras and the SMPL body model.

[Download a PDF of the poster here](https://tinyurl.com/bdctxjku)

[Download my thesis through ProQuest](https://www.proquest.com/docview/2840141315)

Please contact me if you are unable to access the full manuscript: emma.young.r@gmail.com OR emma.r.young@du.edu

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

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/f5073bb3-79f5-4cfc-8d9f-55f9f6790944)


1. "Scan" subject with two azure kinect cameras and physically collect anthropometric measures with a tailor's tape for validation.
   
| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/37651069-8549-4409-8dc1-9bf2df121769)
| :--:
| Camera placement for data collection

| ![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/6d48fa08-811f-46e5-bc21-cdb5aef4b2b0)
| :--:
| Resulting point cloud after two views are combined


2. Recreate missing sections of the subject point cloud (PC).
Originally, the subject PC only includes data of the front and right side. The left side is created by mirroring the right side data. The back is estimated by connecting the back most points on the left and right side. Then, the pointcloud is downsampled. Code for this data processing method is not included here.   

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/afb5db8f-88e4-4102-9929-73d4a46d09c7)

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/3ad577d7-4cf2-4031-a0ec-5da666f1c93d)

![image](https://github.com/EmmaRYoung/PredictionOfBodyShape_SMPL/assets/67296859/bccc00c0-bc33-4fe2-8d7f-9b40e85108e7)



3. Remove vertices from the template SMPL mesh that correspond to incomplete PC data. Removed indices are shown in red on the template male SMPL model below.
Regions of incomplete PC data are the upper back, back of arms, and back of legs. 4136 of the 6890 vertices remain.
![image](https://user-images.githubusercontent.com/67296859/218335092-cf6a6f0b-09e5-4930-8109-83e0a0b8f7b4.png) 


4. Deform the modified SMPL mesh out to the subject point cloud using a deformable CPD algorithm from cycpd. Recover a SMPL model that most closely matches the principal component scores (Betas) of this new instance. 
![image](https://user-images.githubusercontent.com/67296859/218341618-ae6c9b8e-90b9-4036-b5da-f898fabecf4e.png)
![image](https://user-images.githubusercontent.com/67296859/218341147-8025c4ad-e242-4914-abdb-43c79f6b6581.png)


5. Perform a zero-th order optimization to further refine the body model. The objective function uses a KNN algorithm to minimize the distance between the modified SMPL mesh (N=4136) and subject point cloud (N=8272).  

![image](https://user-images.githubusercontent.com/67296859/218504142-8b9c505e-310b-42e1-9ac4-9551f390a3bf.png)

![image](https://user-images.githubusercontent.com/67296859/218503697-42fdf827-8d4f-4e7c-a8ea-3e002066b05e.png)


6. Assess the accuracy of the generated SMPL model. Physical measurements gathered in the data collection process are compared against model generated measurements (shown below) 

![image](https://user-images.githubusercontent.com/67296859/218500862-ca8a0106-dae2-4eae-bcb8-86e467045fc6.png)


# Results
![S02Gif](https://user-images.githubusercontent.com/67296859/218332402-98c949fc-e8ad-4844-a71b-65745d4b6e06.gif) 
![S11Gif](https://user-images.githubusercontent.com/67296859/218332432-0edd0de7-8b55-4e26-bb3a-1f9f71b6c103.gif) 
![S20Gif](https://user-images.githubusercontent.com/67296859/218332629-72865de4-bc88-4301-b0c6-f64767d11ad8.gif)
![S16Gif](https://user-images.githubusercontent.com/67296859/218332677-bff3f9f8-f339-4d18-8f1b-b96dddca54a6.gif)
![S05Gif 1](https://user-images.githubusercontent.com/67296859/218332913-d2bb02e3-c7d7-4d5c-b061-9c60496adec5.gif)
![S06Gif](https://user-images.githubusercontent.com/67296859/218333127-aedc46b3-5035-4a0f-8ba6-bf6774e34685.gif)
![S22Gif](https://user-images.githubusercontent.com/67296859/218332864-4648edcc-d5fa-4eaa-b31e-38d3a80d3038.gif)

