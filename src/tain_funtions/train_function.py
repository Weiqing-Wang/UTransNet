import copy
import os
import time

import torch
from timm.layers import trunc_normal_
from torch import nn
from torch.utils.data import DataLoader

from src.tain_funtions.early_stop import EarlyStopping
from src.tain_funtions.loss_functions import LearnableWeights


def generate_metrics_list(metrics_def):
    list = {}
    for name in metrics_def.keys():
        list[name] = []
    return list

def epoch(scope,train_loader,channels_weights,learnable_weights,training=False):

    model=scope["model"]
    optimizer=scope["optimizer"]
    loss_function=scope["loss_function"]
    metrics_def=scope["metrics_def"]
    scope = copy.copy(scope)
    metrics_list=generate_metrics_list(metrics_def)
    total_loss=0
    if training:
        model.train()
    else:
        model.eval()
    for tensors in train_loader:
        if "device" in scope and scope["device"] is not None:
            tensors=[tensor.to(scope["device"]) for tensor in tensors]
        loss,pre_output=loss_function(model,tensors,channels_weights,learnable_weights)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss+=loss.item()
        scope["loss"]=loss
        scope["batch"] = tensors
        scope["output"] = pre_output
        for name,metric in metrics_def.items():
            value=metric["on_batch"](scope)
            metrics_list[name].append(value)
    scope["metrics_list"] = metrics_list
    metrics = {}
    for name in metrics_def.keys():
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)
    return total_loss,metrics

def train(simulation_directory,scope,train_dataset,test_dataset,patience,after_epoch,channels_weights,scheduler):
    epochs=scope["epochs"]
    model=scope["model"]
    metrics_def=scope["metrics_def"]
    scope=copy.copy(scope)
    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_test_metrics"] = None
    scope["best_test_loss"] = float("inf")
    scope["best_model"] = None
    train_loader=DataLoader(train_dataset,batch_size=scope["batch_size"],shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=scope["batch_size"],shuffle=False)

    if os.path.isfile(os.path.join(simulation_directory, "train_log.txt")):

        os.remove(os.path.join(simulation_directory, "train_log.txt"))
        print("Previous train log deleted successfully")
    else:
        print("Train log does not exist")

    early_stopping=EarlyStopping(patience, verbose=True)
    initial_weights = [1,1,1,0.01,0.001]
    learnable_weights = LearnableWeights(initial_weights)
    for i in range(1,epochs+1):

        scope["epoch_id"] = i
        with open(os.path.join(simulation_directory, "train_log.txt"), "a") as f:
            print("Epoch:",i)
            f.write("Epoch #" + str(i) + "\n")
            scope["dataset"]=train_dataset
            if i==epochs:
                start_time=time.time()
            train_loss, train_metrics = epoch(scope,train_loader,channels_weights,learnable_weights,training=True)
            if i==epochs:
                end_time=time.time()
                print("One epoch time:",end_time-start_time)
                scope['time']=end_time-start_time
            scope["train_loss"] = train_loss
            scope["train_metrics"] = train_metrics
            print("Train Loss = "+str(train_loss),flush=True)
            for name in metrics_def.keys():
                print("Train "+metrics_def[name]["name"]+" = "+str(train_metrics[name]),flush=True)
                f.write("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]) + "\n")
            del scope["dataset"]
            scope["dataset"]=test_dataset
            with torch.no_grad():
                test_loss,test_metrics=epoch(scope,test_loader,channels_weights,learnable_weights,training=False)
            scope["test_loss"]=test_loss
            scope["test_metrics"]=test_metrics
            print("Test Loss = "+str(test_loss),flush=True)
            for name in metrics_def.keys():
                print("Test "+metrics_def[name]["name"]+" = "+str(test_metrics[name]),flush=True)
                f.write("\tTest " + metrics_def[name]["name"] + " = " + str(test_metrics[name]) + "\n")

            del scope["dataset"]
            is_best=None
            if is_best is None:
                is_best = test_loss < scope["best_test_loss"]
            if is_best:
                scope["best_train_metric"] = train_metrics
                scope["best_train_loss"] = train_loss
                scope["best_test_metrics"] = test_metrics
                scope["best_test_loss"] = test_loss
                scope["best_model"] = copy.deepcopy(model)
                print("Model saved!", flush=True)
            if after_epoch is not None:
                after_epoch(scope)
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print("Early stopping", flush=True)
                break
        scheduler.step()
    return (scope["best_model"], scope["best_train_metric"], scope["best_train_loss"],scope["best_test_metrics"],
            scope["best_test_loss"],scope["epoch_id"],scope['time'])


def train_model(simulation_directory,model,loss_function,train_dataset,test_dataset,optimizer,after_epoch,epochs,batch_size,patience,
                device,channels_weights,scheduler):
    scope = {}
    scope["model"] = model
    scope["loss_function"] = loss_function
    scope["train_dataset"] = train_dataset
    scope["test_dataset"] = test_dataset
    scope["optimizer"] = optimizer
    scope["epochs"] = epochs
    scope["batch_size"] = batch_size
    scope["device"] = device
    metrics_def = {
        'mse': {
            'name': 'Total MSE',
            'on_batch': lambda scope: float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
            'on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
        },
        'ux': {
            'name': 'Ux MSE',
            'on_batch': lambda scope: float(
                torch.sum((scope["output"][:, 0, :, :] - scope["batch"][1][:, 0, :, :]) ** 2)),
            'on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
        },
        'uy': {
            'name': 'Uy MSE',
            'on_batch': lambda scope: float(
                torch.sum((scope["output"][:, 1, :, :] - scope["batch"][1][:, 1, :, :]) ** 2)),
            'on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
        },
        'p': {
            'name': 'p MSE',
            'on_batch': lambda scope: float(
                torch.sum((scope["output"][:, 2, :, :] - scope["batch"][1][:, 2, :, :]) ** 2)),
            'on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
        }
    }
    scope["metrics_def"] = metrics_def
    return train(simulation_directory,scope,train_dataset,test_dataset,patience,after_epoch,channels_weights,scheduler)
