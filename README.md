# DistributedMachineLearning_Project

The instance types used for this project is posted below. We used 5 different instances with varying numbers of vCPU and GPU. Distributed machine leraning was abandoned because the scale of our model was not large enough to justify it. We did not see a speedup in performance as the communication overhead was too large and our models had to be adjusted too much to converge properly. 

Instead we used 5 different types of paid instances offered by AWS to test our various types of GPU for machine learning. The same model was used in each instance. We were able to load these instances with helpful AMIs that included all the necessary libraries, including tensorflow, cuda API for nvidia GPUs. Anaconda was also pre-installed in these AMIs, and several environments were provided to choose from. We used "tensorflow2_p38".

# Key results (More results and images in each corresponding instance folder)

         g3s.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla M60) Meant for graphics
Initialization time:
      Training time:
      cost per hour:

          g4.xlarge: 16 vCPU - 1 GPU (Nvidia Tesla M60) Meant for graphics
Initialization time:
      Training time:
      cost per hour:

       g4dn.4xlarge: 16 vCPU - 1 GPU (Nvidia Tesla T4) Meant for ML, but cheaper
Initialization time:
      Training time:
      cost per hour:

          p2.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla K80) 
Initialization time:
      Training time:
      cost per hour:

         p3.2xlarge:  8 vCPU - 1 GPU (Nvidia Tesla V100) Elite GPU
Initialization time:
      Training time:
      cost per hour:

# What to talk about in the paper - some ideas:

1) Discuss main objection - Using machine learning various AWS EC2 instances, using tensflow. Discuss each instance and the type of GPU it has. Can mention which GPUs are meant for ML and which arent.

2) Discussion of kaggle data used for keypoint detection. Information found here.
https://www.kaggle.com/competitions/facial-keypoints-detection/data
Can talk about how the data is preprocessed, total of ~7000 images (code snippets). Some images were missing keypoint labels in the csv, hadd to fill in the values. This affected training accuracy but not too much. Can post some example images from this dataset, what locations we're trying to predict.

3) Discussion of CNN model used, complete description found in code & results , use code snippets. Give model summary (which includes all layers). Can discuss what the layers do.

4) optimizer used, loss function used, and measured metrics

5) discussing of training accuracy

6) pictures of faces printed out with actual keypoints vs predicted keypoints

7) timing results between the instances

8) cost of instances