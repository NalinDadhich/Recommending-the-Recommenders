#!/bin/bash
#ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=weightnorm          #Set the job name to "JobExample4"
#SBATCH --time=00:30:00              #Set the wall clock limit to 14hr and 30min
#SBATCH --ntasks=20                   #Request 1 task
#SBATCH --mem=24000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Logs.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=ALL              #Send email on all job events

#First Executable Line
#cd CNN/My_Directory/layered_compression
module load cuDNN/7.0.5-CUDA-9.0.176
ml Python/3.5.2-intel-2017A
cd ..
source tdenv/bin/activate
cd OpenLearning4DeepRecsys-AC
CUDA_VISIBLE_DEVICES=0,1 python ccf_net_cross.py
