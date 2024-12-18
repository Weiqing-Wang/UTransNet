from torch import nn


class MLP(nn.Module):
    def __init__(self,in_features,hidden_feature,out_features,a_activation=nn.GELU):
        super().__init__()
        out_features=out_features or in_features
        hidden_feature=hidden_feature or in_features
        self.fc1=nn.Linear(in_features,hidden_feature)
        self.act=a_activation()
        self.fc2=nn.Linear(hidden_feature,out_features)
        self.drop=nn.Dropout(0.1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        # x=self.drop(x)
        return x