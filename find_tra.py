import pickle
import torch

# 1. 设置打印阈值
torch.set_printoptions(threshold=float('inf'))
# 1. 读取 pickle 文件
with open("/home/ylc/mpd-public/scripts/train_diffusion/logs/train_diffusion_2025-03-12_10-55-39/dataset_subdir___robot_id___RobotPanda/include_velocity___True/use_ema___True/variance_schedule___exponential/n_diffusion_steps___25/predict_epsilon___True/unet_dim_mults_option___0/0/results_inference/30/results_data_dict.pickle", "rb") as f:
    data = pickle.load(f)

# 2. 打印内容
print("数据类型:", data['trajs_iters'].shape) #torch.Size#([31, 50, 64, 14])
print("内容:", data['trajs_iters'][0][0])
