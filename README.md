# DistributedMachineLearning_Project

The instance types used for this project are posted below. We used 4 different instances with varying numbers of vCPU and GPU. 
Distributed machine learning was experimented with, but we did not get the results we expected. We did not see a speedup in performance as the communication overhead was too large and our models had to be adjusted too much to converge properly. If we used more expensive instances with 100GB/s network, we would have seen a speedup in training time for distributed ML (using many VMs) but it turned out to not be worth. What was done instead is a survey of single VMs, most of which include a GPU.

We used 4 different types of paid instances offered by AWS to test our various types of GPU for machine learning. 
The same model was used in each instance. 
We were able to load these instances with helpful AMIs that included all the necessary libraries, 
including tensorflow, cuda API for nvidia GPUs. 
Anaconda was also pre-installed in these AMIs, and several environments were provided to choose from. 
We used "conda activate tensorflow2_p38".
Unfortunately, we were not allowed to use the top-tier P-type instances (these offer the best GPUs for ML, 
like Nvidia's Tesla V100). 
Details provided below.

## Key results Below (More results and images in each corresponding instance folder)

# c5.4xlarge:  16 vCPU - No GPU

      Initialization time:  18.27s (Didnt have to load GPU libraries)

            Training time:  6663.82s

            Cost per hour:  0.68 USD per Hour

Notes: ML without a GPU is pretty slow!



# g3s.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla M60)

      Initialization time:  23.07s

            Training time:  883.80s

            Cost per hour:  0.75 USD per Hour

Notes: Drastic speedup, even with only 4 vCPUs, from the use of 1 basic GPU that was not built for ML, 
but basic application graphics.



# g3.4xlarge:  16 vCPU - 1 GPU (Nvidia Tesla M60)

      Initialization time:  21.91s

            Training time:  877.64s   

            Cost per hour:  1.14 USD per Hour

Notes: Going from 4 vCPUs to 16 vCPUs did not provide a significant speedup. 
When Tensorflow detects a GPU (and has the appropriate APIs installed) most of the training will take place on the GPU, which is why the additional vCPUs do not help.



# g4dn.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla T4) (Recently added)

      Initialization time:  18.12s

            Training time:  595.34s

            Cost per hour:  0.710 USD per Hour

Notes: The T4 GPU is a datacenter GPU that has ML in mind,
so it's no surprise that we see a speedup in ML training performance, 
despite having half the vCPUs. Note that this instance is cheaper than the previous.



# 2x_g4dn.xlarge:  8 vCPU - 2 GPUs (2x Nvidia Tesla T4) - Up to 25 Gigabit Network (Recently added)

      Initialization time:  17.43s

            Training time:  696.12s (very curious result, we'll talk about this one)

            Cost per hour:  2 * 0.710 USD per Hour

Notes: Distributed Machine Learning with 2 instances - xlarge (less vCPUs)



# 4x_g4dn.xlarge:  16 vCPU - 4 GPUs (4x Nvidia Tesla T4) - Up to 25 Gigabit Network (Recently added)

      Initialization time:  18.62s

            Training time:  438.22s (fast! but the model did not converge nearly as quickly, a lot to talk about here)

            Cost per hour:  4 * 0.710 USD per Hour

Notes: Distributed Machine Learning with 4 instances - 



# p2.xlarge:  4 vCPU - 1 GPU (Nvidia Tesla K80) 

      Initialization time: Abandoned!

            Training time: Abandoned!

            Cost per hour: Abandoned!

Notes: Older datacenter GPU, originally manufacturer in 2014, but still powerful.
Unfortunately, AWS denied the request to use this resource (all P type).
By default, the limit for all P types is set to 0 to prevent new users from accidentally creating these
top-tier instances and ending up with a big bill. You basically have to be a consistent paying customer
in order to be granted access (they take it case by case). I even appealed the denial but ultimately
AWS explained in full detail why they cannot allow access to this resource at this time.



# p3.2xlarge:  8 vCPU - 1 GPU (Nvidia Tesla V100) Elite Datacenter GPU

      Initialization time: Abandoned!

            Training time: Abandoned!

            Cost per hour: Abandoned!

Notes: Top of the line modern datacenter GPU with ML in mind. I wish we got to use this!
See the note above, access to this resource was also denied.


# What to talk about in the paper - some ideas:

1) Discuss main objection - Using machine learning various AWS EC2 instances, using Tensorflow. Discuss each instance and the type of GPU it has. Can mention which GPUs are meant for ML and which arent.

2) Discussion of kaggle data used for keypoint detection. Information found here.
https://www.kaggle.com/competitions/facial-keypoints-detection/data
Can talk about how the data is preprocessed, total of ~7000 images (code snippets). Some images were missing keypoint labels in the csv, hadd to fill in the values. This affected training accuracy but not too much. Can post some example images from this dataset, what locations we're trying to predict.

3) Discussion of CNN model used, complete description found in code & results , use code snippets. Give model summary (which includes all layers). Can discuss what the layers do.

4) optimizer used, loss function used, and measured metrics

5) discussing of training accuracy

6) pictures of faces printed out with actual keypoints vs predicted keypoints

7) timing results between the instances

8) cost of instances

9) Struggles with AWS. They did not grant access to their P-type instances. They aim to keep these instance types at 40-60%
load, and explained that they will usually only grant access for users that have gone through a few billing cycles. This is to ensure
that new users do not end up with unexpected large bills by accident, as these instance types can be expensive.
