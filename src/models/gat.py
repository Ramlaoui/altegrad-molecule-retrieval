import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel


class GATEncoder(nn.Module):
    def __init__(
        self,
        num_node_features,
        nout,
        nhid,
        nheads,
        graph_hidden_channels,
        type_model="encoder",
    ):
        super(GATEncoder, self).__init__()
        self.type_model = type_model
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=nheads)
        self.conv2 = GATConv(
            nheads * graph_hidden_channels, graph_hidden_channels, heads=nheads
        )
        self.conv3 = GATConv(
            nheads * graph_hidden_channels, graph_hidden_channels, heads=1
        )
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
        self.ln1 = nn.LayerNorm((graph_hidden_channels))
        self.ln2 = nn.LayerNorm((nout))

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x_graph = global_mean_pool(x, batch)

        if self.type_model == "qformer":
            batch_node, batch_mask = to_dense_batch(x, batch)
            batch_mask = batch_mask.bool()

            batch_node = torch.cat((x_graph.unsqueeze(1), batch_node), dim=1)
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
            return batch_node, batch_mask, x_graph
        else:
            x = self.mol_hidden1(x_graph).relu()
            x = self.mol_hidden2(x)
            x = self.ln2(x)

            return x


class BaseTextEncoder(nn.Module):
    def __init__(self, model_name, nout, nhid):
        super(BaseTextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.text_proj1 = nn.Linear(768, nhid)
        self.ln1 = nn.LayerNorm((nout))

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        # print(encoded_text.last_hidden_state.size())
        x = encoded_text.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.text_proj1(x)
        x = self.ln1(x)
        return x


class GATModel(nn.Module):
    def __init__(
        self, model_name, num_node_features, nout, nhid, nheads, graph_hidden_channels
    ):
        super(GATModel, self).__init__()
        self.graph_encoder = GATEncoder(
            num_node_features, nout, nhid, nheads, graph_hidden_channels
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
