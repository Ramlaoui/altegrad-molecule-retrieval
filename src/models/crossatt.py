import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import (
    to_dense_batch,
    remove_self_loops,
    add_self_loops,
    softmax,
    degree,
)
from torch_geometric.nn.inits import uniform, glorot, zeros
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoModel


class CrossAttentionModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        nhead,
        nlayers,
        graph_hidden_channels,
        mol_trunc_length=512,
        ninp=768,
        temp=0.07,
        dropout=0.5,
    ):
        super(CrossAttentionModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nhid)
        self.text_hidden2 = nn.Linear(nhid, nout)

        self.nhid = nhid
        self.nout = nout
        self.num_node_features = num_node_features
        self.graph_hidden_channels = graph_hidden_channels
        self.mol_trunc_length = mol_trunc_length

        self.drop = nn.Dropout(p=dropout)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.text_transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        self.temp = nn.Parameter(torch.Tensor([temp]))
        self.register_parameter("temp", self.temp)

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()

        self.conv1 = GCNConv(self.num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

        self.other_params = list(self.parameters())  # get all but bert params

        self.text_transformer_model = AutoModel.from_pretrained(model_name)
        self.text_transformer_model.train()

    def forward(self, graph_batch, input_ids, attention_mask):
        text_encoder_output = self.text_transformer_model(
            input_ids, attention_mask=attention_mask
        )
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x_graph = self.conv3(x, edge_index)
        batch_size = graph_batch.ptr.shape[0] - 1

        # turn pytorch geometric output into the correct format for transformer
        # requires recovering the nodes from each graph into a separate dimension
        node_features = torch.zeros(
            (batch_size, self.mol_trunc_length, self.graph_hidden_channels)
        ).to(batch.device)
        for i, p in enumerate(graph_batch.ptr):
            if p == 0:
                old_p = p
                continue
            node_features[i - 1, : p - old_p, :] = x_graph[
                old_p : torch.min(p, old_p + self.mol_trunc_length), :
            ]
            old_p = p
        node_features = torch.transpose(node_features, 0, 1)

        text_output = self.text_transformer_decoder(
            text_encoder_output["last_hidden_state"].transpose(0, 1),
            node_features,
            tgt_key_padding_mask=attention_mask == 0,
        )

        # Readout layer
        x = global_mean_pool(x_graph, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x)
        x = x.relu()
        x = self.mol_hidden2(x)

        text_x = torch.tanh(self.text_hidden1(text_output[0, :, :]))  # [CLS] pooler
        text_x = self.text_hidden2(text_x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return (text_x, x)

    def forward_text(self, input_ids, attention_mask):
        text_encoder_output = self.text_transformer_model(
            input_ids, attention_mask=attention_mask
        )
        text_x = torch.tanh(
            self.text_hidden1(text_encoder_output.last_hidden_state[:, 0, :])
        )
        text_x = self.text_hidden2(text_x)
        text_x = self.ln2(text_x)
        text_x = text_x * torch.exp(self.temp)
        return text_x

    def forward_graph(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x_graph = global_mean_pool(x, batch)
        x = self.mol_hidden1(x_graph)
        x = x.relu()
        x = self.mol_hidden2(x)
        x = self.ln1(x)
        x = x * torch.exp(self.temp)
        return x

    def get_graph_encoder(self):
        return lambda graph_batch: self.forward_graph(graph_batch)

    def get_text_encoder(self):
        return lambda input_ids, attention_mask: self.forward_text(
            input_ids, attention_mask
        )
