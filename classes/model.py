from torch import nn
import torch



class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(729, 729),
            nn.ReLU(),
            nn.Linear(729, 729),
            nn.ReLU(),
            nn.Linear(729, 729)
        )
        
    def forward(self, x):
        logits = self.network(x).view(-1, 9)

        return logits


class BiggerModel(nn.Module):

    def __init__(self):
        super(BiggerModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(729, 729),
            nn.ReLU(),
            nn.Linear(729, 729),
            nn.ReLU(),
            nn.Linear(729, 729),
            nn.ReLU(),
            nn.Linear(729, 729),
            nn.ReLU(),
            nn.Linear(729, 729)
        )
        
    def forward(self, x):
        logits = self.network(x).view(-1, 9)

        return logits


class DenseModel(nn.Module):

    def __init__(self):
        super(DenseModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(81, 168),
            nn.ReLU(),
            nn.Linear(168, 350),
            nn.ReLU(),
            nn.Linear(350, 729)
        )
        
    def forward(self, x):
        logits = self.network(x).view(-1, 9)

        return logits


class BigDenseModel(nn.Module):

    def __init__(self):
        super(BigDenseModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(81, 162),
            nn.ReLU(),
            nn.Linear(162, 243),
            nn.ReLU(),
            nn.Linear(243, 324),
            nn.ReLU(),
            nn.Linear(324, 405),
            nn.ReLU(),
            nn.Linear(405, 486),
            nn.ReLU(),
            nn.Linear(486, 567),
            nn.ReLU(),
            nn.Linear(567, 648),
            nn.ReLU(),
            nn.Linear(648, 729)
        )
        
    def forward(self, x):
        logits = self.network(x).view(-1, 9)

        return logits
    

class Model81x81(nn.Module):

    def __init__(self):
        super(Model81x81, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(81, 243),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
            nn.Linear(243, 81)
        )
        
    def forward(self, x):
        logits = self.network(x)

        return logits
    


class RecurrentModel(nn.Module):

    def __init__(self):
        super(RecurrentModel, self).__init__()

        self.cell_predict = nn.Sequential(
            nn.Linear(324, 243),
            nn.ReLU(),
            nn.Linear(243, 162),
            nn.ReLU(),
            nn.Linear(162, 81),
        )

        self.cell_update = nn.Sequential(
            nn.Linear(405, 324),
            nn.ReLU(),
            nn.Linear(324, 243),
            nn.ReLU(),
            nn.Linear(243, 162),
            nn.ReLU(),
            nn.Linear(162, 81),
            nn.ReLU(),
            nn.Linear(81, 9)
        )

    def forward(self, x):
        # x.shape = (batch_size, 81, 4)

        z = x.view(-1, 324)
        z = self.cell_predict(z)
        x = torch.cat((x, z.unsqueeze(2)), dim=2)

        x = self.cell_update(x.view(-1, 405))

        return z, x


