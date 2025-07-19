import torch
import torch.nn as nn


class Node_encoder(nn.Module):
    def __init__(self, emb_size = 32, pre_emb_size = 4, layers = 2, reduction = "mean"):
        super().__init__()
        self.emb_size = emb_size * 2
        self.pre_emb_size = pre_emb_size
        self.layers = layers
        self.reduction = reduction

        self.node_row_preencoder = nn.LSTM(1, self.pre_emb_size, num_layers=self.layers, bidirectional=True, batch_first=True)
        self.node_col_encoder_rnn = nn.LSTM(self.pre_emb_size * 2, self.emb_size, num_layers=self.layers, bidirectional=True, batch_first=True)
        self.node_col_encoder_linear = nn.Sequential(
            nn.Linear(2 * (self.emb_size * 2 * self.layers) + self.emb_size, 4 * self.emb_size),
            nn.ReLU(),
            nn.Linear(4 * self.emb_size, 2 * self.emb_size),
            nn.ReLU(),
            nn.Linear(2 * self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size//2),#new
            nn.ReLU(),
            nn.Linear(self.emb_size//2, self.emb_size),#new
            nn.Sigmoid()
        )

    def forward(self, input):                                                                     # input: [batch x num_node x num_node]
        batch, nodes_num, nums = input.size()
        preencoded_row, _ = self.node_row_preencoder(input.view(batch * nodes_num, nodes_num, 1))   # preencoded_row: [batch * num_node x num_node x 2 * emb_size]

        preencoded_row = preencoded_row.view(batch, nodes_num, nodes_num, 2, self.pre_emb_size)
        preencoded_row = preencoded_row.max(dim=-2)[0] if self.reduction == "max" else preencoded_row.mean(dim=-2)

        preencoded_row = preencoded_row.view(batch, nodes_num, nodes_num, self.pre_emb_size)        # preencoded_row: [batch x num_node x num_node x emb_size]

        encoded_col_list = []
        for i in range(nodes_num):
            node_col = preencoded_row[:, :, i]

            not_node_col = preencoded_row[:, :, :i].sum(dim=-2) + preencoded_row[:, :, i + 1:].sum(dim=-2)
            # node_col , not_node_col : [batch, num_node, emb_size]

            o, (h_n, c_n) = self.node_col_encoder_rnn(torch.cat([node_col, not_node_col], dim = -1))  # o: [batch x emb_size]

            if self.reduction == 'max':
                o_reduction = o.view(batch, nodes_num, 2, self.emb_size).max(dim=-2)[0].max(dim=-2)[0]
            else:
                o_reduction = o.view(batch, nodes_num, 2, self.emb_size).mean(dim=-2).mean(dim=-2)

            h_n = h_n.view(batch, -1)  #[batch, 2 * layers * emb_size]
            c_n = c_n.view(batch, -1)  #[batch, 2 * layers * emb_size]

            encoded_col = self.node_col_encoder_linear(torch.cat([h_n, c_n, o_reduction], dim=-1)) 

            encoded_col_list.append(encoded_col)
        col_embeddings = torch.stack(encoded_col_list, dim=-2)        # col_embeddings: [batch x num_node x emb_size]
        col_embeddings, mu, logvar = self.reparameterize(col_embeddings)

        return col_embeddings, mu, logvar

    def encode_rows(self, input, rol_embeddings):
        batch, num_r, num_c = input.size()
        _, __, emb_size = rol_embeddings.size()
        oa_embeddings = torch.stack([input] * emb_size, dim=-1) * rol_embeddings.view(batch, 1, num_c, emb_size)

        node_embeddings = oa_embeddings.max(dim=-2)[0]

        return node_embeddings
        
    def reparameterize(self, h):  
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

class Node_decoder(nn.Module):

    def __init__(self, emb_size=16):
        super().__init__()
        self.emb_size = emb_size

        self.node_decoder = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size * 2),
            nn.ReLU(),

            nn.Linear(self.emb_size * 2, self.emb_size * 2),
            nn.ReLU(),

            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),

            nn.Linear(self.emb_size, 1),
            nn.Sigmoid()
        )

    def forward(self, col_embeddings, node_embeddings):                 # [batch x num_node x emb_size]
        batch, num_col, _ = node_embeddings.size()
        batch, num_row, _ = node_embeddings.size()

        node_embeddings_ = node_embeddings.expand(num_col, batch, num_row, self.emb_size).permute(1, 2, 0, 3)
        col_embeddings_ = col_embeddings.expand(num_row, batch, num_col, self.emb_size).transpose(0, 1)

        oa_embeddings = torch.cat([col_embeddings_, node_embeddings_], dim=-1) # [batch x num_row x num_row x (2 * emb_size)]

        pred = self.node_decoder(oa_embeddings).squeeze(-1)         # pred: [batch x num_row x num_row] 移除最后一个维度

        return pred

class SimilarityPredictor(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.predictor = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1),
            nn.Sigmoid()
        )

    def forward(self, a_embeddings):
        batch, num_a, emb_size = a_embeddings.size()
        # a_embeddings: [batch x num_a x emb_size]
        a_embeddings_stacked = a_embeddings.expand(num_a, batch, num_a, emb_size).transpose(0, 1).float() # [batch x num_a x num_a' x emb_size]
        # exchange attribute and stack dimensions
        a_embeddings_stacked_transposed = a_embeddings_stacked.transpose(1, 2) # [batch x num_a' x num_a x emb_size]

        x = torch.cat([a_embeddings_stacked, a_embeddings_stacked_transposed], dim = -1) # [batch x num_a' x num_a x 2 * emb_size]

        pred = self.predictor(x) # [batch x num_a' x num_a x 1]
        pred = pred.view(pred.size()[:-1]) # [batch x num_a' x num_a]

        return pred

class LengthPredictor(nn.Module):
    def __init__(self, emb_size, reduction="mean"):
        super().__init__()
        self.emb_size = emb_size
        self.reduction = reduction
        self.predictor = nn.Sequential(
            nn.Linear(self.emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
            )
    def forward(self, a_embeddings):
        # a_embeddings: [batch x num_a x emb_size]
        if self.reduction == "mean":
            reduced = a_embeddings.mean(dim=-2) # [batch x emb_size]
        else:
            reduced = a_embeddings.max(dim=-2)[0] # [batch x emb_size]
        pred = self.predictor(reduced) # [batch x 1]

        return pred.view(-1)