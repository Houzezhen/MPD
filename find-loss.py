import numpy as np
train_loss = np.load('/home/ylc/mpd-public/scripts/train_diffusion/logs/train_diffusion_2025-03-14_21-24-20/dataset_subdir___robot_id___RobotUr10/include_velocity___True/use_ema___True/variance_schedule___exponential/n_diffusion_steps___25/predict_epsilon___True/unet_dim_mults_option___0/0/checkpoints/train_losses.npy', allow_pickle=True)
print(train_loss)  # 输出应为1维数组（如(1000,)表示1000个损失值）
