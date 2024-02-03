import torch
from torch_geometric.nn import GMMConv
from torch_geometric.nn import global_mean_pool
from torch import nn
import torch.nn.functional as F
from dgl.nn import JumpingKnowledge
from transformers import AutoModel


class MonetGraph(nn.Module):
    def __init__(
        self, num_node_features, nout, nhid, nlayers, graph_hidden_channels, kernel_size
    ):
        super(MonetGraph, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.jk = JumpingKnowledge(
            mode="lstm", in_feats=graph_hidden_channels, num_layers=nlayers
        )  # Jumping Knowledge layer
        for i in range(nlayers):
            if i == 0:
                n_input = num_node_features
            else:
                n_input = graph_hidden_channels
            self.bns.append(nn.BatchNorm1d(graph_hidden_channels))
            self.layers.append(
                GMMConv(
                    in_channels=n_input,  # Number of input features
                    out_channels=graph_hidden_channels,  # Number of output features
                    dim=2,  # Dimensionality of pseudo-coordinate
                    kernel_size=kernel_size,  # Number of kernels
                )
            )

        self.projection1 = nn.LazyLinear(nhid)
        self.projection2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm((graph_hidden_channels))
        self.ln2 = nn.LayerNorm((nout))

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch

        h_list = [x]
        for i, layer in enumerate(self.layers):
            h = layer(h_list[-1], edge_index, edge_attr)
            h = self.bns[i](h)  # Use the corresponding batch normalization layer
            if i == len(self.layers) - 1:
                h = self.dropout(h)
            else:
                h = self.dropout(self.relu(h))
            h_list.append(h)

        node_representation = self.jk(
            h_list
        )  # Use Jumping Knowledge to aggregate representations

        h_graph = global_mean_pool(node_representation, batch)

        x = self.projection1(h_graph).relu()
        x = self.dropout(x)
        x = self.projection2(x)
        x = self.ln2(x)

        return x


class BaseTextEncoder(nn.Module):
    def __init__(self, model_name, nout, nhid):
        super(BaseTextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(768, nhid)
        self.linear2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm((nout))

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        # print(encoded_text.last_hidden_state.size())
        x = encoded_text.last_hidden_state[:, 0, :]
        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.ln1(x)
        return x


class Monet(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        nlayers,
        graph_hidden_channels,
        kernel_size,
    ):
        super(Monet, self).__init__()
        self.graph_encoder = MonetGraph(
            num_node_features, nout, nhid, nlayers, graph_hidden_channels, kernel_size
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
