import torch
import torch.nn as nn


class NumberUpperBoundPredictor(nn.Module):

    def __init__(self, emb_size, reduction="mean"):
        super().__init__()
        self.emb_size = emb_size
        self.reduction = reduction
        self.predictor = nn.Sequential(
            nn.Linear(self.emb_size + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, node_embeddings, metric_predictor_output, attrs):

        attr_node_embeddings = torch.cat((node_embeddings, attrs), dim=-1)
        reduced = attr_node_embeddings.mean(dim=-2) 
        input = torch.cat([reduced, metric_predictor_output.view(-1, 1)], dim=-1)
        pred = self.predictor(input)  # [batch x 1]

        return pred.view(-1)

