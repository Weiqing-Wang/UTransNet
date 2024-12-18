import torch


def optimizer(model,learning_rate,weight_decay):
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    return optimizer
