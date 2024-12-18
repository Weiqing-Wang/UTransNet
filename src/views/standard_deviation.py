import json
import matplotlib.pyplot as plt
import numpy as np

# 设置字体大小
plt.rcParams.update({'font.size': 14})

# 存储四种架构的结果列表
results_UAutoED_list = []
results_UAutoEDRes_list = []
results_UMultiED_list = []
results_UTransNet_list = []

# 遍历读取四种架构的每个文件数据
for i in range(5):
    file_path_UAutoED = f'../../result/Single_noRes_conv_{i + 1}_1000.json'
    file_path_UAutoEDRes = f'../../result/Single_Res_conv_{i + 1}_1000.json'
    file_path_UMultiED = f'../../result/Multi_noRes_conv_{i + 1}_1000.json'
    file_path_UTransNet = f'../../result/UTransNet_conv_{i + 1}_1000.json'

    with open(file_path_UAutoED, 'r') as f_UAutoED:
        results_UAutoED = json.load(f_UAutoED)
        results_UAutoED_list.append(results_UAutoED)

    with open(file_path_UAutoEDRes, 'r') as f_UAutoEDRes:
        results_UAutoEDRes = json.load(f_UAutoEDRes)
        results_UAutoEDRes_list.append(results_UAutoEDRes)

    with open(file_path_UMultiED, 'r') as f_UMultiED:
        results_UMultiED = json.load(f_UMultiED)
        results_UMultiED_list.append(results_UMultiED)

    with open(file_path_UTransNet, 'r') as f_UTransNet:
        results_UTransNet = json.load(f_UTransNet)
        results_UTransNet_list.append(results_UTransNet)

# 存储四种架构的各个损失曲线数据
UAutoED_mse_loss_curve_all = []
UAutoED_ux_loss_curve_all = []
UAutoED_uy_loss_curve_all = []
UAutoED_p_loss_curve_all = []

UAutoEDRes_mse_loss_curve_all = []
UAutoEDRes_ux_loss_curve_all = []
UAutoEDRes_uy_loss_curve_all = []
UAutoEDRes_p_loss_curve_all = []

UMultiED_mse_loss_curve_all = []
UMultiED_ux_loss_curve_all = []
UMultiED_uy_loss_curve_all = []
UMultiED_p_loss_curve_all = []

UTransNet_mse_loss_curve_all = []
UTransNet_ux_loss_curve_all = []
UTransNet_uy_loss_curve_all = []
UTransNet_p_loss_curve_all = []

# 提取每种架构每个文件中的损失曲线数据
for results_ae in results_UAutoED_list:
    UAutoED_mse_loss_curve_all.append(results_ae['curves']['test_mse_curve'])
    UAutoED_ux_loss_curve_all.append(results_ae['curves']['test_ux_curve'])
    UAutoED_uy_loss_curve_all.append(results_ae['curves']['test_uy_curve'])
    UAutoED_p_loss_curve_all.append(results_ae['curves']['test_p_curve'])

for results_ae in results_UAutoEDRes_list:
    UAutoEDRes_mse_loss_curve_all.append(results_ae['curves']['test_mse_curve'])
    UAutoEDRes_ux_loss_curve_all.append(results_ae['curves']['test_ux_curve'])
    UAutoEDRes_uy_loss_curve_all.append(results_ae['curves']['test_uy_curve'])
    UAutoEDRes_p_loss_curve_all.append(results_ae['curves']['test_p_curve'])

for results_ae in results_UMultiED_list:
    UMultiED_mse_loss_curve_all.append(results_ae['curves']['test_mse_curve'])
    UMultiED_ux_loss_curve_all.append(results_ae['curves']['test_ux_curve'])
    UMultiED_uy_loss_curve_all.append(results_ae['curves']['test_uy_curve'])
    UMultiED_p_loss_curve_all.append(results_ae['curves']['test_p_curve'])

for results_ae in results_UTransNet_list:
    UTransNet_mse_loss_curve_all.append(results_ae['curves']['test_mse_curve'])
    UTransNet_ux_loss_curve_all.append(results_ae['curves']['test_ux_curve'])
    UTransNet_uy_loss_curve_all.append(results_ae['curves']['test_uy_curve'])
    UTransNet_p_loss_curve_all.append(results_ae['curves']['test_p_curve'])



# 生成 epoch 列表
epochs = list(range(1, 1001))

# 计算四种架构的total MSE标准差
UAutoED_mse_loss_curve_std = [np.std(x) for x in zip(*UAutoED_mse_loss_curve_all)]
UAutoEDRes_mse_loss_curve_std = [np.std(x) for x in zip(*UAutoEDRes_mse_loss_curve_all)]
UMultiED_mse_loss_curve_std = [np.std(x) for x in zip(*UMultiED_mse_loss_curve_all)]
UTransNet_mse_loss_curve_std = [np.std(x) for x in zip(*UTransNet_mse_loss_curve_all)]

# 绘制四种架构的total MSE标准差曲线
plt.figure()
plt.plot(epochs, UAutoED_mse_loss_curve_std, label='UAutoED', color='#FF9999')
plt.plot(epochs, UAutoEDRes_mse_loss_curve_std, label='UAutoEDRes', color='#0072B2')
plt.plot(epochs, UMultiED_mse_loss_curve_std, label='UMultiED', color='#009E73')
plt.plot(epochs, UTransNet_mse_loss_curve_std, label='UTransNet', color='#FF0000')
# plt.title('Total MSE Loss Standard Deviation Curves')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Total')
plt.xlim(0, 1000)
plt.ylim(0, 0.8)
plt.savefig('total_mse_loss_std_curves.eps', dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

# 计算四种架构的Ux标准差
UAutoED_ux_loss_curve_std = [np.std(x) for x in zip(*UAutoED_ux_loss_curve_all)]
UAutoEDRes_ux_loss_curve_std = [np.std(x) for x in zip(*UAutoEDRes_ux_loss_curve_all)]
UMultiED_ux_loss_curve_std = [np.std(x) for x in zip(*UMultiED_ux_loss_curve_all)]
UTransNet_ux_loss_curve_std = [np.std(x) for x in zip(*UTransNet_ux_loss_curve_all)]

# 绘制四种架构的Ux标准差曲线
plt.figure()
plt.plot(epochs, UAutoED_ux_loss_curve_std, label='UAutoED', color='#FF9999')
plt.plot(epochs, UAutoEDRes_ux_loss_curve_std, label='UAutoEDRes', color='#0072B2')
plt.plot(epochs, UMultiED_ux_loss_curve_std, label='UMultiED', color='#009E73')
plt.plot(epochs, UTransNet_ux_loss_curve_std, label='UTransNet', color='#FF0000')
# plt.title('Ux MSE Loss Standard Deviation Curves')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Ux')
plt.xlim(0, 1000)
plt.ylim(0, 0.6)
plt.savefig('ux_mse_loss_std_curves.eps', dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

# 计算四种架构的Uy标准差
UAutoED_uy_loss_curve_std = [np.std(x) for x in zip(*UAutoED_uy_loss_curve_all)]
UAutoEDRes_uy_loss_curve_std = [np.std(x) for x in zip(*UAutoEDRes_uy_loss_curve_all)]
UMultiED_uy_loss_curve_std = [np.std(x) for x in zip(*UMultiED_uy_loss_curve_all)]
UTransNet_uy_loss_curve_std = [np.std(x) for x in zip(*UTransNet_uy_loss_curve_all)]

# 绘制四种架构的Uy标准差曲线
plt.figure()
plt.plot(epochs, UAutoED_uy_loss_curve_std, label='UAutoED', color='#FF9999')
plt.plot(epochs, UAutoEDRes_uy_loss_curve_std, label='UAutoEDRes', color='#0072B2')
plt.plot(epochs, UMultiED_uy_loss_curve_std, label='UMultiED', color='#009E73')
plt.plot(epochs, UTransNet_uy_loss_curve_std, label='UTransNet', color='#FF0000')
# plt.title('Uy MSE Loss Standard Deviation Curves')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Uy')
plt.xlim(0, 1000)
plt.ylim(0, 0.2)
plt.savefig('uy_mse_loss_std_curves.eps', dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

# 计算四种架构的P标准差
UAutoED_p_loss_curve_std = [np.std(x) for x in zip(*UAutoED_p_loss_curve_all)]
UAutoEDRes_p_loss_curve_std = [np.std(x) for x in zip(*UAutoEDRes_p_loss_curve_all)]
UMultiED_p_loss_curve_std = [np.std(x) for x in zip(*UMultiED_p_loss_curve_all)]
UTransNet_p_loss_curve_std = [np.std(x) for x in zip(*UTransNet_p_loss_curve_all)]

# 绘制四种架构的P标准差曲线
plt.figure()
plt.plot(epochs, UAutoED_p_loss_curve_std, label='UAutoED', color='#FF9999')
plt.plot(epochs, UAutoEDRes_p_loss_curve_std, label='UAutoEDRes', color='#0072B2')
plt.plot(epochs, UMultiED_p_loss_curve_std, label='UMultiED', color='#009E73')
plt.plot(epochs, UTransNet_p_loss_curve_std, label='UTransNet', color='#FF0000')
# plt.title('P MSE Loss Standard Deviation Curves')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('P')
plt.xlim(0, 1000)
plt.ylim(0, 0.4)
plt.savefig('p_mse_loss_std_curves.eps', dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()