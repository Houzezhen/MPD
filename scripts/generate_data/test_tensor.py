import os

import torch


def load_and_check(file_path):
    try:
        # 加载数据
        loaded_data = torch.load(file_path)

        # 提取张量（假设文件存储单个张量）
        if not isinstance(loaded_data, torch.Tensor):
            raise ValueError("文件内容非张量类型")

        # 空张量检测
        is_empty = (loaded_data.numel() == 0) or (0 in loaded_data.shape)
        print(f"张量形状：{loaded_data.shape}，是否为空：{is_empty}")
        if is_empty:
            try:
                current_dir = os.path.dirname(file_path)
                os.rmdir(current_dir)
                print(f"空文件夹 '{current_dir}' 已删除")
            except FileNotFoundError:
                print(f"错误：路径 '{current_dir}' 不存在")
        return loaded_data
    except FileNotFoundError:
        print("错误：文件路径无效")
    except Exception as e:
        print(f"数据加载异常：{str(e)}")

for i in range(0,499):
    load_and_check(f"/home/ylc/mpd-public/scripts/generate_data/logs/generate_trajectories_2025-04-09_10-07-22/env_id___EnvSpheres3D/robot_id___RobotPandaCar/{i}/trajs-free.pt")
#loaded_data = torch.load("/home/ylc/mpd-public/scripts/generate_data/logs/generate_trajectories_2025-03-31_00-18-12/env_id___EnvSpheres3D/robot_id___RobotPandaCar/36/trajs-free.pt")
#print(loaded_data)