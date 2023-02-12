# PredictionOfBodyShape_SMPL
Thesis (in progress) work to develop a noninvasive way of predicting a subject's body shape using data from Azure Kinect depth cameras and the SMPL body model.
View my ORS poster presentation here: YOUTUBE
Download a PDF of the poster here: tinyurl.com/bdctxjku

# Acknowledgements 
This project was supported in part by a DU Professional Research Opportunities for Faculty grant
Thank you to my colleagues for their expert advice and support:
Peter Laz, Casey Myers, Paul Rullkoetter, Thor Andreassen
This project would not be possible without Abigail Eustace's thesis about the Azure Kinect camera. Thank you for your hard work

#Requirements
The SMPL body model and its requirements: https://smpl.is.tue.mpg.de/
cycpd: https://github.com/gattia/cycpd

# Workflow
Currently, I use a multiphase optimization process to align the SMPL body with personalized Kinect data. The first two phases take advantage of the similarities between the SMPL and the Kinect skeleton to roughly position the SMPL model inside the Kinect depth data. A registry of similar limb lengths and joints is created for the Kinect and SMPL skeletons (KRL,SRL and SRJ, KRJ respectively)

1. Minimize the distance between limb lengths

$$f_1(\beta) = \sum_{i=1}^n (SRL_i - KRL_i)^2 \ \ \ variables = 10,\  N = 13$$ 


2. Minimize the distance between similar joints 

$$f_1(\theta) = \sum_{i=1}^n ||SRJ_i - KRJ_i|| \ \ \ variables = 10,\  N = 13$$ 

In the next phases, the objective function minimizes the distance between vertices on the SMPL mesh (6890 nodes) and their nearest neighbor on the point cloud. Phase 3 and 4 position the surface mesh for phase 5 where the steepest descent algorithm is left to run for many iterations.

3. K-nearest neighbor algorithm - Variables = 10

4. K-nearest neighbor algorithm - Variables = 63

5. K-nearest neighbor algorithm - Variables = 10

The whole process takes ~18 minutes

# Results
<img src="https://user-images.githubusercontent.com/67296859/199391821-00779a20-cbfb-4d04-8540-2b883a23a33e.png" width="1100" />
<img src="https://user-images.githubusercontent.com/67296859/199391939-54bef88a-fa19-404f-9b1e-69cb9107d00d.png" width="1100" />



# Examples of data collected with the Azure Kinect depth camera

<img src="https://user-images.githubusercontent.com/67296859/199389362-049263f3-2968-405c-8164-fcf5de022ec6.png" width="450" /> <img src="https://user-images.githubusercontent.com/67296859/199389371-3bee53d4-e424-4887-a6bf-4b07d6912905.png" width="315" />

<img src="https://user-images.githubusercontent.com/67296859/199389385-0709755f-a6ff-48f6-ae9f-5c3723141323.png" width="450" /> <img src="https://user-images.githubusercontent.com/67296859/199389393-c28185b6-a8f5-4d68-a58b-3818ba4845f3.png" width="315" />

<img src="https://user-images.githubusercontent.com/67296859/199389196-8524e932-70b2-4790-aa51-7a5441efa9f5.png" width="450" /> <img src="https://user-images.githubusercontent.com/67296859/199389230-0afaec2d-d9d7-4e4f-95aa-6f10d71b9e4f.png" width="260" />
