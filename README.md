# DistributedMachineLearning_Project

The instance types used for this project are posted below. We used 4 different instances with varying numbers of vCPU and GPU. 
Distributed machine learning was experimented with, but we did not get the results we expected. We did not see a speedup in performance as the communication overhead was too large and our models had to be adjusted too much to converge properly. If we used more expensive instances with 100GB/s network, we would have seen a speedup in training time for distributed ML (using many VMs) but it turned out to not be worth. What was done instead is a survey of single VMs, most of which include a GPU.


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


