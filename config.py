'''
 # @ Author: Meet Patel
 # @ Create Time: 2024-11-07 09:19:00
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-11-09 09:08:16
 # @ Description:
 '''


# Model Architecture
input_nc = 3
output_nc = 3
input_hw = 256
output_hw = 256
base_channels_gen = 64
base_channels_dis = 48

# Dataset
dataset_path = "/mnt/d/d/DeepLearning/Workfoster-CGAN/Dense_Haze_NTIRE19/"
random_flip = True
sample_size = [input_hw, input_hw]

# Training
num_epochs = 10
batch_size = 1
vgg_layers_to_extract = [4, 9, 16, 23]
normalize_gram_matrix = True