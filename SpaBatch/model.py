import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm


def sce_fun(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

# def sce_loss(x, y, t=2):
#     x = F.normalize(x, p=2, dim=-1)
#     y = F.normalize(y, p=2, dim=-1)
#     cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
#     loss = -torch.log(cos_m.pow_(t))
#     return loss.mean()


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )



class VGAE_model(nn.Module):
    def __init__(
            self,
            input_dim,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_cluster_n=20,
    ):
        super(VGAE_model, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_cluster_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2

        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.decoder = GCNConv(self.latent_dim, self.input_dim)

        self.gc1 = Sequential('x, edge_index', [
            (GCNConv(self.feat_hidden2, self.gcn_hidden1), 'x, edge_index -> x1'),
            BatchNorm(self.gcn_hidden1),
            nn.ReLU(inplace=True),
        ])
        self.gc2 = Sequential('x, edge_index', [
            (GCNConv(self.gcn_hidden1, self.gcn_hidden2), 'x, edge_index -> x1'),
            BatchNorm(self.gcn_hidden2),
            nn.ReLU(inplace=True),
        ])
        self.gc3 = Sequential('x, edge_index', [
            (GCNConv(self.gcn_hidden1, self.gcn_hidden2), 'x, edge_index -> x1'),
            BatchNorm(self.gcn_hidden2),
            nn.ReLU(inplace=True),
        ])

        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))

        self.dc = InnerProductDecoder(p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.gcn_hidden2 + self.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(
            self,
            target
    ):
        weight = (target ** 2) / torch.sum(target, 0)  # 每个元素除以总和，归一化
        return (weight.t() / torch.sum(weight, 1)).t()


    def loss(
        self,
        decoded,
        x,
        preds,
        labels,
        mu,
        logvar,
        n_nodes,
        norm,
        x_init,
        x_rec,
        sce_weight=10,
        bce_kld_weight=0.1,
        ):
        #这一部分是求x和解码器解码后得到的decoded之间的误差
        #mse_fun = torch.nn.MSELoss()
        #mse_loss = mse_fun(decoded, x)
        sce_loss = sce_fun(x_rec, x_init)

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
              1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return sce_weight * sce_loss + bce_kld_weight * (bce_logits_loss + KLD)

    def encoding_mask_noise(self, x, adj, mask_rate=0.2):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device) # 随机排列节点索引

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        #out_x[mask_nodes] = 0.0
        #out_x[token_nodes] = torch.zeros_like(out_x[token_nodes])

        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()

        return out_x, use_adj, (mask_nodes, keep_nodes)

    def forward(
        self,
        x,
        adj,
        mask_rate=0.2,
        ):
        x, adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, adj, mask_rate=mask_rate)
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z, adj)

        recon = de_feat.clone()
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        #软分布
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z, x_init, x_rec


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(
        self,
        dropout,
        act=torch.sigmoid,
        ):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(
        self,
        z,
        ):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj





