# DistributedMachineLearning_Project

The instance types used for this project is posted below. We used 5 different instances with varying numbers of vCPU and GPU. Distributed machine leraning was abandoned because the scale of our model was not large enough to justify it. We did not see a speedup in performance as the communication overhead was too large and our models had to be adjusted too much to converge properly. 

Instead we used 5 different types of paid instances offered by AWS to test our various types of GPU for machine learning. The same model was used in each instance. We were able to load these instances with helpful AMIs that included all the necessary libraries, including tensorflow, cuda API for nvidia GPUs. Anaconda was also pre-installed in these AMIs, and several environments were provided to choose from. We used "conda activate tensorflow2_p38".

# Key results (More results and images in each corresponding instance folder)

# c5.4xlarge:  16 vCPU - No GPU

      Initialization time:  18.27s (Didnt have to load GPU libraries)

      Training time:  

      cost per hour:  0.68 USD per Hour

Notes: ML without a GPU is pretty slow!



# g3s.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla M60)

      Initialization time:  23.07s

      Training time:  883.80s

      cost per hour:  0.75 USD per Hour

Notes: Drastic speedup, even with only 4 vCPUs, from the use of 1 basic GPU that was not built for ML, but basic application graphics.



# g3.4xlarge:  16 vCPU - 1 GPU (Nvidia Tesla M60)

      Initialization time:  21.91s

      Training time:  877.64s   

      cost per hour:  1.14 USD per Hour

Notes: Going from 4 vCPUs to 16 vCPUs did not provide a signifigant speedup. When Tensorflow detects a GPU (and has the appropriate APIs installed) most of the training will take place on the GPU, which is why the additional vCPUs do not help.



# g4dn.2xlarge:  8 vCPU - 1 GPU (Nvidia Tesla T4)

      Initialization time:  19.45s

      Training time:  700.49s

      cost per hour:  0.752 USD per Hour

Notes: The T4 GPU is a datacenter GPU that has ML in mind, so it's no surprise that we see a speedup in ML training performance, despite having half the vCPUs. Note that this instance is cheaper than the previous.



# p2.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla K80) 

      Initialization time:

      Training time:

      cost per hour:

Notes: Older datacenter GPU, originally manufacturer in 2014, but still powerful.



# p3.2xlarge:  8 vCPU - 1 GPU (Nvidia Tesla V100) Elite Datacenter GPU

      Initialization time:

      Training time:

      cost per hour:

Notes: Top of the line modern datacenter GPU with ML in mind.

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