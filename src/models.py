import torch.nn as nn
import torch
import os
import math
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch_geometric.nn.conv import MessagePassing

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = self.conv1(x, edge_index, edge_type, edge_norm)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return score, F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

class BERTSeqClf(nn.Module):
    def __init__(self, num_labels, model='bert-base', \
                n_layers_freeze=0, wiki_model='', \
                n_layers_freeze_wiki=0, dim_graph_features=100, \
                senti_flag=False, graph_flag=False, senti_len=4):
        super(BERTSeqClf, self).__init__()

        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoModel
        if model == 'bert-base':
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        elif model == 'bertweet':
            self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        else:  # covid-twitter-bert
            self.bert = AutoModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

        n_layers = 12 if model != 'covid-twitter-bert' else 24
        
        if n_layers_freeze > 0:
            n_layers_ft = n_layers - n_layers_freeze
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        if wiki_model:
            if wiki_model == model:
                self.bert_wiki = self.bert
            else:  # bert-base
                self.bert_wiki = AutoModel.from_pretrained('bert-base-uncased')

            n_layers = 12
            if n_layers_freeze_wiki > 0:
                n_layers_ft = n_layers - n_layers_freeze_wiki
                for param in self.bert_wiki.parameters():
                    param.requires_grad = False
                for param in self.bert_wiki.pooler.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_layers_ft, -1):
                    for param in self.bert_wiki.encoder.layer[i].parameters():
                        param.requires_grad = True
        if senti_flag:
            senti_bert = AutoModel.from_pretrained("twitter-roberta-base-sentiment-latest")
            self.senti_bert = senti_bert
            senti_config = self.senti_bert.config
            self.senti_dropout = nn.Dropout(senti_config.hidden_dropout_prob)
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if wiki_model and wiki_model != model:
            hidden = config.hidden_size + self.bert_wiki.config.hidden_size
        elif senti_flag:
            hidden = config.hidden_size + senti_config.hidden_size
        else:
            hidden = config.hidden_size
        if graph_flag:
            hidden = hidden + dim_graph_features
        self.senti_flag = senti_flag
        self.graph_flag = graph_flag
        self.classifier = nn.Linear(hidden, num_labels)
        self.senti_classifier = nn.Linear((senti_len+num_labels), num_labels)
        self.model = model
        self.MC = np.load("data/distribution.npy")
    
    def maximizerF(X, self):
        MC = self.MC.copy()
        X = 1+X
        MC = 1+MC
        Xnew = (X+MC[0]+2) + (X+MC[1]+2) + (X+MC[2]+2)
        d = (0.5 * np.average(cdist(X, MC), axis=1).reshape([len(X), 1]))
        Xnew = (Xnew + d) / (X)
        return Xnew
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                input_ids_wiki=None, attention_mask_wiki=None, graph_features=None, senti_features=None, 
                input_ids_senti=None, attention_mask_senti=None):
        bert_outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)

        pooled_bert_output = bert_outputs.pooler_output
        pooled_bert_output = self.dropout(pooled_bert_output)
        if input_ids_wiki is not None:
            outputs_wiki = self.bert_wiki(input_ids_wiki, attention_mask=attention_mask_wiki)
            pooled_wiki_output = outputs_wiki.pooler_output
            pooled_wiki_output = self.dropout(pooled_wiki_output)
            pooled_bert_output = torch.cat((pooled_bert_output, pooled_wiki_output), dim=1)
        if self.senti_flag:
            senti_bert_output = self.senti_bert(input_ids=input_ids_senti, attention_mask=attention_mask_senti)
            pooled_senti_output = senti_bert_output.pooler_output
            pooled_senti_output = self.senti_dropout(pooled_senti_output)
            pooled_bert_output = torch.cat((pooled_bert_output, pooled_senti_output), dim=1)
        if self.graph_flag:
            concated_output = torch.cat((pooled_bert_output, graph_features), dim=1)
        else:
            concated_output = pooled_bert_output
        
        logits = self.classifier(concated_output)
        return logits, concated_output
