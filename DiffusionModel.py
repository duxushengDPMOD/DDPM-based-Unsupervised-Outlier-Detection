import torch.nn as nn
class DiffusionModel(nn.Module):
    def __init__(self, args, train_x):
        super(DiffusionModel, self).__init__()
        self.features = train_x.shape[1]
        self.num_steps = args.num_steps
        self.num_units = args.units

        self.linears = nn.ModuleList(
            [
                nn.Linear(self.features, self.num_units),# 0
                nn.ReLU(),# 1
                nn.Linear(self.num_units, self.num_units),# 2
                nn.ReLU(),# 3
                nn.Linear(self.num_units, self.num_units),# 4
                nn.ReLU(),# 5
                nn.Linear(self.num_units, self.features),# 6
            ]
        ).to(args.device)

        # embedding the num steps
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.num_steps, self.num_units),# 0
                nn.Embedding(self.num_steps, self.num_units),# 1
                nn.Embedding(self.num_steps, self.num_units),# 2
            ]
        ).to(args.device)

    def forward(self, x, t):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            t_embedding_cpu=t_embedding.cpu().detach().numpy()
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x