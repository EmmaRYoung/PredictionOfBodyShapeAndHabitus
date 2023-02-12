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
1. "Scan" subject with two azure kinect cameras and physically collect extensive anthropometric measures with a tailor's tape for validation. 


2. Recreate missing sections of the subject point cloud.

3. Remove vertices from the template SMPL mesh that correspond to incomplete point cloud data

4. Deform the modified SMPL mesh out to the subject point cloud. Recover a SMPL model that most closely matches this new instance. 

5. Perform a zero-th order optimization to further refine the body model. The objective function uses a KNN algorithm to minimize the distance between the SMPL mesh and subject point cloud.  

6. Assess the accuracy of the generated SMPL model

# Results
![S02Gif](https://user-images.githubusercontent.com/67296859/218332402-98c949fc-e8ad-4844-a71b-65745d4b6e06.gif) 
![S11Gif](https://user-images.githubusercontent.com/67296859/218332432-0edd0de7-8b55-4e26-bb3a-1f9f71b6c103.gif) 
![S20Gif](https://user-images.githubusercontent.com/67296859/218332629-72865de4-bc88-4301-b0c6-f64767d11ad8.gif)
![S16Gif](https://user-images.githubusercontent.com/67296859/218332677-bff3f9f8-f339-4d18-8f1b-b96dddca54a6.gif)
![S05Gif 1](https://user-images.githubusercontent.com/67296859/218332913-d2bb02e3-c7d7-4d5c-b061-9c60496adec5.gif)
![S22Gif](https://user-images.githubusercontent.com/67296859/218332864-4648edcc-d5fa-4eaa-b31e-38d3a80d3038.gif)

