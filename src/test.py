# import configparser
# import os
# import pickle
# import time
# import numpy as np
# import torch
#
#
# # 定义一个函数用于统计给定样本的预测时间相关指标
# def calculate_prediction_metrics(sample_x, model, num_predictions):
#     times = []
#     for t in range(num_predictions):
#         time1 = time.time()
#         model(sample_x.to('cuda'))
#         time2 = time.time()
#         elapsed_time = time2 - time1
#         times.append(elapsed_time)
#     average_time = np.mean(times)
#     times=times[1:]
#     std_time = np.std(times)
#     return average_time, std_time
#
#
# # 读取配置文件
# config = configparser.ConfigParser()
# config.read("../config/config.ini")
#
# # 获取测试数据集路径
# test_dataset_path = os.path.join(config["parameters"]["data_path"], "n_test_dataset.pkl")
#
# # 加载测试数据集
# test_dataset = pickle.load(open(test_dataset_path, 'rb'))
# x, y = test_dataset[:]


#
# # 加载模型
# model = torch.load('../result/UTransNet_conv_2_1000.pt')
# model.to('cuda')
# model.eval()
#
# # 选择要进行测试的样本索引（这里选择索引为10的样本，可按需更改）
# sample_index = 10
# test_x = x[sample_index: sample_index + 1]
#
# average_time, std_time = calculate_prediction_metrics(test_x, model, 1001)
# print(average_time, std_time)

import configparser
import os
import pickle
import time
import numpy as np
import torch

# 读取配置文件
config = configparser.ConfigParser()
config.read("../config/config.ini")

# 获取测试数据集路径
test_dataset_path = os.path.join(config["parameters"]["data_path"], "n_test_dataset.pkl")

# 加载测试数据集
test_dataset = pickle.load(open(test_dataset_path, 'rb'))
x, y = test_dataset[:]

# 加载模型
model = torch.load('../result/UTransNet_conv_2_1000.pt')
model.to('cuda')
model.eval()

# 提取10个样本数据
test_x = x[:100]

# 用于存储每次预测10个样本的总时间
times = []
# 设定预测次数为1001次，去除第一次可能的热身时间（如果有的话），实际有效次数为1000次
num_predictions = 1001
for _ in range(num_predictions):
    time1 = time.time()
    pre_out = model(test_x.to('cuda'))
    time2 = time.time()
    elapsed_time = (time2 - time1)/100
    times.append(elapsed_time)
times = times[1:]

# times列表存储的是每次预测10个样本整体所花费的时间。
# 计算一个样本平均的预测时间（总时间除以样本数量和预测次数）
average_time_per_sample = np.mean(times)
# 计算标准差，这里的标准差是关于每次预测10个样本整体时间的标准差，不应该除以10
std_time = np.std(times)

print(f"Average prediction time per sample over {num_predictions - 1} predictions: {average_time_per_sample}")
print(f"Standard deviation of prediction time over {num_predictions - 1} predictions: {std_time}")