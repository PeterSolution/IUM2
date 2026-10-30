from torch import nn

class Model(nn.Module):
    def __init__(self,input, MidSize, output):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input, MidSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(MidSize, output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    