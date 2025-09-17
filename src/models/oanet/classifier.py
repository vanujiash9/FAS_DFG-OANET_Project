import torch.nn as nn

class OANetClassifier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        return self.fc(x)