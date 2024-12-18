import configparser
import json
import os
import pickle

import torch.cuda
from torch import nn

from src.models.UTransNet import UTransNet
from src.tain_funtions.loss_functions import loss_function
from src.tain_funtions.optimizer import optimizer
from src.tain_funtions.train_function import train_model
from src.utils.tools import setup_seed



#read the model's parameters
config=configparser.ConfigParser()
config.read('../config/config.ini')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u_activation=nn.LeakyReLU
a_activation=nn.LeakyReLU
a_norm=nn.LayerNorm

in_channels=int(config['parameters']['in_channels'])
out_channels=int(config['parameters']['out_channels'])
filters=[int(i) for i in config['parameters']['filters'].split(',')]
kernel_size=int(config['parameters']['kernel_size'])
# padding=int(config['parameters']['padding'])
depths=int(config['parameters']['depths'])
mlp_ratio=int(config['parameters']['mlp_ratio'])
multi_out_channels=int(config['parameters']['multi_out_channels'])
res_type=int(config['parameters']['res_type'])
if res_type == 1:
    res_type = True
else:
    res_type = False
if multi_out_channels==1:
    multi_out_channels=True
else:
    multi_out_channels=False

learning_rate=float(config['parameters']['learning_rate'])
weight_decay=float(config['parameters']['weight_decay'])
epochs=int(config['parameters']['epochs'])
patiences=int(config['parameters']['patiences'])
batch_size=int(config['parameters']['batch_size'])
c_layers=int(config['parameters']['c_layers'])
d_layers=int(config['parameters']['d_layers'])
num_heads=int(config['parameters']['num_heads'])
data_path=config['parameters']['data_path']
save_path=config['parameters']['save_path']

train_dataset_path=os.path.join(config["parameters"]["data_path"], "n_train_dataset.pkl")
test_dataset_path=os.path.join(config["parameters"]["data_path"], "n_test_dataset.pkl")
simulation_directory = config["parameters"]["save_path"]
atten_type=int(config['parameters']['atten_type'])
if atten_type==1:
    atten_type='convolution'
elif atten_type==0:
    atten_type='linear'

if not os.path.exists(simulation_directory):
    os.mkdir(simulation_directory)

train_dataset=pickle.load(open(train_dataset_path,'rb'))
test_dataset=pickle.load(open(test_dataset_path,'rb'))

setup_seed(1)

loss_func=loss_function

model=UTransNet(in_channels,
                out_channels,
                filters,
                kernel_size,
                num_heads,
                mlp_ratio,
                depths,
                a_activation,
                a_norm,
                u_activation,
                multi_out_channels,
                atten_type,
                res_type)

model.to(device)

opt=optimizer(model,learning_rate,weight_decay)
scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=40,gamma=0.9)

config = {}
train_loss_curve = []
test_loss_curve = []
train_mse_curve = []
test_mse_curve = []
train_ux_curve = []
test_ux_curve = []
train_uy_curve = []
test_uy_curve = []
train_p_curve = []
test_p_curve = []


def after_epoch(scope):
    train_loss_curve.append(scope["train_loss"])
    test_loss_curve.append(scope["test_loss"])
    train_mse_curve.append(scope["train_metrics"]["mse"])
    test_mse_curve.append(scope["test_metrics"]["mse"])
    train_ux_curve.append(scope["train_metrics"]["ux"])
    test_ux_curve.append(scope["test_metrics"]["ux"])
    train_uy_curve.append(scope["train_metrics"]["uy"])
    test_uy_curve.append(scope["test_metrics"]["uy"])
    train_p_curve.append(scope["train_metrics"]["p"])
    test_p_curve.append(scope["test_metrics"]["p"])
channels_weights=None
best_model, best_train_loss,best_train_metric, best_test_loss, best_test_metrics, epoch_id,time=train_model(simulation_directory,
                                                                                                       model,
                                                                                                       loss_func,
                                                                                                       train_dataset,
                                                                                                       test_dataset,
                                                                                                       opt,
                                                                                                       after_epoch,
                                                                                                       epochs,
                                                                                                       batch_size,
                                                                                                       patiences,
                                                                                                       device,
                                                                                                       channels_weights,
                                                                                                       scheduler)

metrics = {}
metrics['time']= time
metrics["best_train_metric"] = best_train_metric
metrics["best_train_loss"] = best_train_loss
metrics["best_test_metrics"] = best_test_metrics
metrics["best_test_loss"] = best_test_loss
curves = {}
curves["train_loss_curve"] = train_loss_curve
curves["test_loss_curve"] = test_loss_curve
curves["train_mse_curve"] = train_mse_curve
curves["test_mse_curve"] = test_mse_curve
curves["train_ux_curve"] = train_ux_curve
curves["test_ux_curve"] = test_ux_curve
curves["train_uy_curve"] = train_uy_curve
curves["test_uy_curve"] = test_uy_curve
curves["train_p_curve"] = train_p_curve
curves["test_p_curve"] = test_p_curve
config["metrics"] = metrics
config["curves"] = curves

torch.save(best_model,os.path.join(simulation_directory, "NEW_Multi_noRes_conv_2_" + str(epoch_id) + ".pt"))
with open('../result/' + "NEW_Multi_noRes_conv_2_"+str(epoch_id)+".json", "w") as file:
    json.dump(config, file)
print(best_train_loss)
print(best_train_metric)
print(best_test_loss)
print(best_test_metrics)


