import json
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
with open('../../result/Multi_noRes_conv_2_1000.json','r') as f_ae:
    results_ae=json.load(f_ae)
ae_test_loss_curve=results_ae['curves']['test_loss_curve']
ae_test_mse_loss_curve=results_ae['curves']['test_mse_curve']
ae_test_ux_loss_curve=results_ae['curves']['test_ux_curve']
ae_test_uy_loss_curve=results_ae['curves']['test_uy_curve']
ae_test_p_loss_curve=results_ae['curves']['test_p_curve']

epochs_ae=list(range(1,1001))

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 第一张子图：全部MSE损失
axs[0, 0].plot(epochs_ae, ae_test_mse_loss_curve[:], label='AE', color='b')

axs[0, 0].set_title('Total MSE Loss Curves')
axs[0, 0].legend()
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('MSE Loss')
axs[0, 0].set_xlim(0, 1000)
axs[0, 0].set_ylim(0, 10)

# 第二张子图：全部ux损失
axs[0, 1].plot(epochs_ae, ae_test_ux_loss_curve[:], label='AE', color='b')

axs[0, 1].set_title('Ux MSE Loss Curves')
axs[0, 1].legend()
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Ux Loss')
axs[0, 1].set_xlim(0, 1000)
axs[0, 1].set_ylim(0, 10)

# 第三张子图：全部uy损失
axs[1, 0].plot(epochs_ae, ae_test_uy_loss_curve[:], label='AE', color='b')

axs[1, 0].set_title('Uy MSE Loss Curves')
axs[1, 0].legend()
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Uy Loss')
axs[1, 0].set_xlim(0, 1000)
axs[1, 0].set_ylim(0, 2)
# 第四张子图：全部p损失
axs[1, 1].plot(epochs_ae, ae_test_p_loss_curve[:], label='AE', color='b')

axs[1, 1].set_title('P MSE Loss Curves')
axs[1, 1].legend()
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('P Loss')
axs[1, 1].set_xlim(0, 1000)
axs[1, 1].set_ylim(0, 2)

# 调整子图间距
plt.tight_layout()

plt.savefig('mse_curves.eps', dpi=1000, bbox_inches='tight',pad_inches=0)
# 显示图形
plt.show()