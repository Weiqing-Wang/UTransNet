import torch


class LearnableWeights:
    def __init__(self, initial_weights):
        self.weights = torch.nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32), requires_grad=True)
    def get_weights(self):
        return self.weights

def loss_function(model,batch,channels_weights,learnable_weights):
    x, y = batch
    out = model(x)
    lossU = torch.abs((out[:, 0, :, :] - y[:, 0, :, :])).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    lossV = torch.abs((out[:, 1, :, :] - y[:, 1, :, :])).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    lossP = torch.abs((out[:, 2, :, :] - y[:, 2, :, :])).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    weights = learnable_weights.get_weights()
    loss = (weights[0] * lossU + weights[1] * lossV + weights[2] * lossP)

    p_error = torch.abs((out[:, 2, :, :] - y[:, 2, :, :]))
    epsilon_p=weights[3]
    beta_p=weights[4]
    p_penalty = beta_p * torch.max(torch.zeros_like(p_error), p_error - epsilon_p)
    p_penalty = p_penalty.reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    loss=loss+p_penalty

    return torch.sum(loss), out


def loss_function1(model,batch,channels_weights,learnable_weights):
    x, y = batch
    out = model(x)
    print(out.shape)
    lossU = ((out[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    lossV = ((out[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    lossP = torch.abs((out[:, 2, :, :] - y[:, 2, :, :])).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
    loss = (lossU + lossV + lossP)/channels_weights
    return torch.sum(loss), out