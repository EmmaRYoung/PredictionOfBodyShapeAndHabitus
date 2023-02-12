# PredictionOfBodyShape_SMPL
Thesis (in progress) work to develop a noninvasive way of predicting a subject's body shape using data from Azure Kinect depth cameras and the SMPL body model.

View a digital version of my Orthopaedic Research Society 2023 poster presentation here: YOUTUBE

[Download a PDF of the poster here](https://tinyurl.com/bdctxjku)

# Acknowledgements 
This project was supported in part by a University of Denver Professional Research Opportunities for Faculty grant

Thank you to my colleagues for their expert advice and support:

Peter Laz, Casey Myers, Paul Rullkoetter, Thor Andreassen

This project would not be possible without Abigail Eustace's thesis about the Azure Kinect camera. Thank you for your hard work!

# Requirements
[The SMPL body model and its requirements](https://smpl.is.tue.mpg.de/)

[cycpd](https://github.com/gattia/cycpd)

# Workflow
1. "Scan" subject with two azure kinect cameras and physically collect anthropometric measures with a tailor's tape for validation. 
<img src="https://user-images.githubusercontent.com/67296859/218335637-c1b7cd65-74be-480b-ba8d-bef8fc3f2924.png" width=80% height=80%> 

<img src="https://user-images.githubusercontent.com/67296859/218335721-773a28ee-151d-4f4f-898f-1d0a5e3b2c97.png"> <img src="https://user-images.githubusercontent.com/67296859/218336082-f8533844-a78c-4c27-a1dd-93a7a89dd167.png">

2. Recreate missing sections of the subject point cloud (PC).
Originally, the subject PC only includes data of the front and right side. The left side is created by mirroring the right side data. The back is estimated by connecting the back most points on the left and right side. Then, the pointcloud is downsampled. Code for this data processing method is not included here.   

![S01PC_FrontRight](https://user-images.githubusercontent.com/67296859/218339779-504152bc-6263-45f4-8fa3-55f54e3a942d.gif)


3. Remove vertices from the template SMPL mesh that correspond to incomplete PC data. Removed indices are shown in red on the template male SMPL model below.
Regions of incomplete PC data are the upper back, back of arms, and back of legs. 4136 of the 6890 vertices remain.
![image](https://user-images.githubusercontent.com/67296859/218335092-cf6a6f0b-09e5-4930-8109-83e0a0b8f7b4.png) 


4. Deform the modified SMPL mesh out to the subject point cloud using a deformable CPD algorithm from cycpd. Recover a SMPL model that most closely matches the principal component scores (Betas) of this new instance. 
![image](https://user-images.githubusercontent.com/67296859/218341618-ae6c9b8e-90b9-4036-b5da-f898fabecf4e.png)
![image](https://user-images.githubusercontent.com/67296859/218341147-8025c4ad-e242-4914-abdb-43c79f6b6581.png)


5. Perform a zero-th order optimization to further refine the body model. The objective function uses a KNN algorithm to minimize the distance between the SMPL mesh and subject point cloud.  

6. Assess the accuracy of the generated SMPL model

# Results
![S02Gif](https://user-images.githubusercontent.com/67296859/218332402-98c949fc-e8ad-4844-a71b-65745d4b6e06.gif) 
![S11Gif](https://user-images.githubusercontent.com/67296859/218332432-0edd0de7-8b55-4e26-bb3a-1f9f71b6c103.gif) 
![S20Gif](https://user-images.githubusercontent.com/67296859/218332629-72865de4-bc88-4301-b0c6-f64767d11ad8.gif)
![S16Gif](https://user-images.githubusercontent.com/67296859/218332677-bff3f9f8-f339-4d18-8f1b-b96dddca54a6.gif)
![S05Gif 1](https://user-images.githubusercontent.com/67296859/218332913-d2bb02e3-c7d7-4d5c-b061-9c60496adec5.gif)
![S06Gif](https://user-images.githubusercontent.com/67296859/218333127-aedc46b3-5035-4a0f-8ba6-bf6774e34685.gif)
![S22Gif](https://user-images.githubusercontent.com/67296859/218332864-4648edcc-d5fa-4eaa-b31e-38d3a80d3038.gif)

