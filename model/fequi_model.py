import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, value_size, num_heads=1, sigmoid_weights=False):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.sigmoid_weights = sigmoid_weights

        self.key_to_queries = nn.Linear(key_size, value_size * num_heads)
        self.value_proj = nn.Linear(value_size, value_size)
        self.softmax = nn.Softmax(dim=-2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, key, values):
        B, L, _ = values.shape            # batch, seq_len, value_size
        queries = self.key_to_queries(key).view(B, self.num_heads, self.value_size)  
        proj = self.value_proj(values)    

        logits = torch.einsum("blv,bhv->blh", proj, queries)

        if self.sigmoid_weights:
            attn = self.softmax(self.sigmoid(logits))
        else:
            attn = self.softmax(logits)   # [B, L, H]

        out = torch.einsum("blh,blv->bhv", attn, proj)

        return out.transpose(1, 2).contiguous().view(B, self.value_size, self.num_heads), logits


class FairEquiGenerator(nn.Module):
    def __init__(self, emb_size, extent_size, hidden_size, num_layers=1, heads=1, dropout=0.1, reduction="mean"):
        super().__init__()
        self.reduction = reduction

        self.num_layers = num_layers
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.extent_size = extent_size
        self.key_size = self.num_layers * self.hidden_size * 2
        self.heads = heads
        self.lstm_input_size = self.extent_size + 2 + 2 * hidden_size  

        def make_mlp(input_size, layers):
            layers_list = []
            for out_size in layers:
                layers_list.append(nn.Linear(input_size, out_size))
                layers_list.append(nn.ReLU())
                input_size = out_size
            return nn.Sequential(*layers_list)

        self.attrnodes_dan = make_mlp(self.emb_size + 10, [128, 128, 128, 64, 64, 64, self.extent_size])
        self.attrnodes_dan.add_module('final_activation', nn.Sigmoid())
        self.lstm = nn.LSTM(self.lstm_input_size, self.hidden_size, num_layers=num_layers, batch_first=True)

        self.self_attention = MultiHeadAttention(self.key_size, self.hidden_size, num_heads=2, sigmoid_weights=True)

        self.lstm_output = nn.Sequential(
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(lambda m: torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu') if hasattr(m, 'weight') else None)

    def forward(self, node_emb, predicted_fconcept_number, attrs, max_fconcept_number=None):

       # self.lstm.flatten_parameters()
       batch, max_node= node_emb.size(0), node_emb.size(1)

       if max_fconcept_number is None:
           max_fconcept_number = predicted_fconcept_number.max()[0] 
       attrnode_emb = torch.cat((node_emb, attrs), dim=-1)

       reduced = attrnode_emb.mean(dim=-2) if self.reduction == "mean" else attrnode_emb.max(dim=-2)[0]

       a_summary = self.attrnodes_dan(reduced).expand(max_fconcept_number, batch, -1).transpose(1, 0)

       indices = torch.arange(max_fconcept_number, device=a_summary.device).float().expand(batch,
                              max_fconcept_number).view(batch, max_fconcept_number, 1)


       predicted_fconcept_number = predicted_fconcept_number.expand(1, max_fconcept_number, batch).transpose(0, -1)
       a_summary = torch.cat([indices, predicted_fconcept_number, a_summary], dim=-1)  

       hidden_and_cell = (torch.zeros(self.num_layers, batch, self.hidden_size, device=a_summary.device),
                          torch.zeros(self.num_layers, batch, self.hidden_size, device=a_summary.device))

       fconcepts_emb = []
       s_attention = torch.zeros([batch, 2, self.hidden_size])

       for i in range(max_fconcept_number): 
           # compute attention
           key = torch.cat([x.transpose(1, 0).reshape(batch, -1) for x in hidden_and_cell], dim=-1)
           if i > 0:
               s_attention, weights = self.self_attention(key, torch.cat(fconcepts_emb, dim=-2))

           s_attention = s_attention.view(batch, self.hidden_size * 2)
           input = torch.cat([s_attention, a_summary[:, i]], dim = -1).unsqueeze(1)                
           output, hidden_and_cell = self.lstm(input, hidden_and_cell)     
           fconcepts_emb.append(output)

       # fconcepts = torch.cat(fconcepts_emb, dim=-2)
       results = []
       for i in range(max_fconcept_number):  
           output = fconcepts_emb[i]                   
           output = output.repeat(1, max_node, 1)       

           fconcept , _ = self.lstm_output[0](output)
           fconcept = self.lstm_output[1](fconcept)
           fconcept = self.lstm_output[2](fconcept)     

           results.append(fconcept)

       return torch.cat(results, dim=-1).permute(0, 2, 1)


