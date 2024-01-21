import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from src.models.bert_blip import BertLMHeadModel, BertConfig
from src.models.gin import GINEncoder
from src.models.gat import GATEncoder


class QFormer(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        nlayers,
        graph_hidden_channels,
        num_query_token,
        graph_encoder_type="gin",
        cross_attention_freq=2,
        temperature=0.07,
    ):
        super(QFormer, self).__init__()
        self.gtm, self.lm = True, True
        encoder_config = BertConfig.from_pretrained(model_name)
        encoder_config.encoder_width = graph_hidden_channels
        # encoder_config.encoder_hidden_states = graph_hidden_channels
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        # encoder_config.is_decoder = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        self.qformer = BertLMHeadModel.from_pretrained(
            model_name, config=encoder_config
        )
        # self.qformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.qformer.state_dict()
        for name, param in self.qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        if graph_encoder_type == "gin":
            self.graph_encoder = GINEncoder(
                num_node_features,
                nout,
                nhid,
                nlayers,
                graph_hidden_channels,
                type_model="qformer",
            )
            # self.ln_graph = nn.LayerNorm((num_node_features))
        elif graph_encoder_type == "gat":
            self.graph_encoder = GATEncoder(
                num_node_features,
                nout,
                nhid,
                nlayers,
                graph_hidden_channels,
                type_model="qformer",
            )

        self.graph_proj = nn.Linear(self.qformer.config.hidden_size, nout)
        self.text_proj = nn.Linear(self.qformer.config.hidden_size, nout)

        self.gtm_head = nn.Linear(self.qformer.config.hidden_size, 2)
        self.temperature = temperature

    def contrast(self, features_graph, features_text, return_sim=False):
        """
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        """
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (
            features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)
        ).squeeze()  # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1)  # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(
            batch_size, dtype=torch.long, device=features_graph.device
        )
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def forward(self, graph_batch, input_ids, attention_mask):
        batch_node, batch_mask, h_graph = self.graph_encoder(graph_batch)
        batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        # batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,  # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(
            query_output.last_hidden_state
        )  # shape = [B, num_q, D]
        text_output = self.qformer.bert(
            input_ids, attention_mask=attention_mask, return_dict=True
        )  # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        sim_g2t, sim_t2g, loss_gtc = self.contrast(
            graph_feats, text_feats, return_sim=True
        )

        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            g_emb = batch_node
            g_mask = batch_mask
            text_ids = input_ids.clone()
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb[neg_idx])
                graph_mask_neg.append(g_mask[neg_idx])

            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids[neg_idx])
                text_atts_neg.append(attention_mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_ids, text_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [attention_mask, attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(
                query_tokens_itm.size()[:-1], dtype=torch.long, device=input_ids.device
            )
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat(
                [g_emb, graph_embeds_neg, g_emb], dim=0
            )  # pos, neg, pos
            graph_atts_all = torch.cat([g_mask, graph_mask_neg, g_mask], dim=0)

            output_itm = self.qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[
                :, : query_tokens_itm.size(1), :
            ]  # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [
                    torch.ones(batch_size, dtype=torch.long),
                    torch.zeros(2 * batch_size, dtype=torch.long),
                ],
                dim=0,
            ).to(input_ids.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        return loss_gtc, loss_gtm

    def graph_forward(self, graph_batch):
        batch_node, batch_mask, h_graph = self.graph_encoder(graph_batch)
        batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,  # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state)
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats

    def text_forward(self, input_ids, attention_mask):
        text_output = self.qformer.bert(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        return text_feats
