import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel


class GINEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        nhid,
        nlayers,
        graph_hidden_channels,
        skip_connection="last",
        type_model="encoder",
    ):
        super(GINEncoder, self).__init__()
        self.skip_connection = skip_connection
        self.type_model = type_model
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(nlayers):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(graph_hidden_channels, 2 * graph_hidden_channels),
                    nn.ReLU(),
                    nn.Linear(2 * graph_hidden_channels, graph_hidden_channels),
                )
            )
            self.bns.append(nn.BatchNorm1d(graph_hidden_channels))
            self.layers.append(
                GINConv(
                    self.mlps[i],
                    train_eps=False,
                )
            )

        self.projection1 = nn.Linear(graph_hidden_channels, nhid)
        self.projection2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm((graph_hidden_channels))
        self.ln2 = nn.LayerNorm((nout))

        # self.gin = GIN(
        #     num_node_features,
        #     hidden_channels=graph_hidden_channels,
        #     out_channels=nout,
        #     num_layers=nlayers,
        #     dropout=0.1,
        # )

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        h_list = [x]
        for i, layer in enumerate(self.layers):
            h = layer(h_list[-1], edge_index)
            h = self.bns[i](h)
            if i == len(self.layers) - 1:
                h = self.dropout(h)
            else:
                h = self.dropout(self.relu(h))
            h_list.append(h)

        if self.skip_connection == "last":
            node_representation = h_list[-1]
        elif self.skip_connection == "sum":
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        h_graph = global_mean_pool(node_representation, batch)

        if self.type_model == "qformer":
            batch_node, batch_mask = to_dense_batch(node_representation, batch)
            batch_mask = batch_mask.bool()

            batch_node = torch.cat((h_graph.unsqueeze(1), batch_node), dim=1)
            batch_mask = torch.cat(
                (
                    torch.ones(
                        (batch_mask.shape[0], 1), dtype=torch.bool, device=batch.device
                    ),
                    batch_mask,
                ),
                dim=1,
            )

            batch_node = self.ln1(batch_node)
            return batch_node, batch_mask, h_graph
        else:
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


class GINMol(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, nlayers, graph_hidden_channels
    ):
        super(GINMol, self).__init__()
        self.graph_encoder = GINEncoder(
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
