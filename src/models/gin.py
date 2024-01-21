from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel


class BaseGraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, nlayers, graph_hidden_channels):
        super(BaseGraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.gin = GIN(
            num_node_features,
            hidden_channels=graph_hidden_channels,
            out_channels=nout,
            num_layers=nlayers,
            dropout=0.1,
        )

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.gin(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.ln(x)
        return x


class BaseTextEncoder(nn.Module):
    def __init__(self, model_name, nout, nhid):
        super(BaseTextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(768, nhid)
        self.linear2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        # print(encoded_text.last_hidden_state.size())
        x = encoded_text.last_hidden_state[:, 0, :]
        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GINMol(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, nlayers, graph_hidden_channels
    ):
        super(GINMol, self).__init__()
        self.graph_encoder = BaseGraphEncoder(
            num_node_features, nout, nhid, nlayers, graph_hidden_channels
        )
        self.text_encoder = BaseTextEncoder(model_name, nout, nhid)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder
