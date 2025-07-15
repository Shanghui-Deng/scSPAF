import math
import sys
from DAE_ZINB import DAE_ZINB
from DAE_Ber import DAE_Ber
import torch.nn
import numpy as np
import opt
from encoder import *


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)
        x = self.output_layer(x)
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, view, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(view, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, view)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = x.unsqueeze(1)
        return x


class MLP(nn.Module):
    def __init__(self, z_emb_size1, dropout_rate):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_emb_size1, z_emb_size1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, z_x, z_y):
        q_x = self.mlp(z_x)
        q_y = self.mlp(z_y)
        return q_x, q_y

class SelfAttention(nn.Module):
    def __init__(self, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        queries = q
        keys = k
        values = v
        n, d = queries.shape
        scores = torch.mm(queries, keys.t()) / math.sqrt(d)
        att_weights = F.softmax(scores, dim=1)
        att_emb = torch.mm(self.dropout(att_weights), values)
        return att_weights, att_emb

def adaptiveGFM(hs,view,adaptive_weight,attention_net,p_net,p_sample):
    hs_tensor = torch.tensor([]).cuda()
    for v in range(view):
        hs_tensor = torch.cat((hs_tensor, torch.mean(hs[v], 1).unsqueeze(1)), 1)  # d * v
    # transpose
    hs_tensor = hs_tensor.t()
    # process by the attention
    hs_atten = attention_net(hs_tensor, hs_tensor, hs_tensor)  # v * 1
    # learn the view sampling distribution
    p_learn = p_net(p_sample)  # v * 1
    # regulatory factor
    r = hs_atten * p_learn
    s_p = nn.Softmax(dim=0)
    r = s_p(r)
    # adjust adaptive weight
    adaptive_weight = r * adaptive_weight
    # obtain fusion feature
    fusion_feature = torch.zeros([hs[0].shape[0], hs[0].shape[1]]).cuda()
    for v in range(view):
        fusion_feature = fusion_feature + adaptive_weight[v].item() * hs[v]
    return fusion_feature


class scSPAF(nn.Module):
    def __init__(self, ae1, ae2, gae1, gae2, n_node=None):
        super(scSPAF, self).__init__()
        self.ae1 = ae1
        self.ae2 = ae2
        self.gae1 = gae1
        self.gae2 = gae2
        self.attlayer1 = SelfAttention(dropout=0.1)
        self.attlayer2 = SelfAttention(dropout=0.1)
        self.mlp = MLP(opt.args.n_z, dropout_rate=0.1)
        self.fc = nn.Linear(opt.args.n_z + opt.args.n_z, opt.args.n_z)
        self.attention_net = MultiHeadAttention(n_node, opt.args.attention_dropout_rate, opt.args.num_heads,
                                                opt.args.attn_bias_dim)
        self.p_net = FeedForwardNetwork(3, opt.args.ffn_size, opt.args.attention_dropout_rate)
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5),
                           requires_grad=True)  # Z_ae, Z_igae
        self.alpha = Parameter(torch.zeros(1))  # ZG, ZL
        self.cluster_centers1 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(n_node, opt.args.n_clusters),
            nn.Softmax(dim=1)
        )
        self.DAE_ZINB = DAE_ZINB(opt.args.n_d1, opt.args.ae_n_enc_1, opt.args.ae_n_enc_2, opt.args.n_z,
                                 opt.args.dropout)
        self.DAE_Ber = DAE_Ber(opt.args.n_d2, opt.args.ae_n_enc_1, opt.args.ae_n_enc_2, opt.args.n_z, opt.args.dropout)

    def emb_fusion(self, adj, z_ae, z_igae):
        z_i = self.a * z_ae + (1 - self.a) * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.alpha * z_g + z_l
        return z_tilde

    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / 1.0
        num = torch.pow(1.0 + num, -(1.0 + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def forward(self, x1, adj1, x2, adj2, p_sample, adaptive_weight, pretrain=False):
        # node embedding encoded by AE
        zx = self.ae1.encoder(x1)
        zy = self.ae2.encoder(x2)
        # node embedding encoded by IGAE
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)
        zx_weights, z_gx = self.attlayer1(z_igae1, z_igae1, zx)
        zy_weights, z_gy = self.attlayer2(z_igae2, z_igae2, zy)
        q_x, q_y = self.mlp(zx, zy)
        cl_loss = crossview_contrastive_Loss(q_x, q_y)
        emb_con = torch.cat([q_x, q_y], dim=1)
        z_xy = self.fc(emb_con)
        zg = adaptiveGFM(hs=[z_gx, z_gy, z_xy], view=3, adaptive_weight=adaptive_weight,
                         attention_net=self.attention_net, p_net=self.p_net, p_sample=p_sample)

        # decoder for DAE_ZINB
        latent_zinb = self.DAE_ZINB.fc_decoder(zg)
        normalized_x_zinb = self.DAE_ZINB.decoder_scale(latent_zinb)
        Final_x_zinb = torch.sigmoid(normalized_x_zinb)

        # decoder for DAE_Ber
        latent_ber = self.DAE_Ber.fc_decoder(zg)
        recon_x_ber = self.DAE_Ber.decoder_scale(latent_ber)
        Final_x_ber = torch.sigmoid(recon_x_ber)

        # AE decoding
        x_hat1 = self.ae1.decoder(z_gx)
        x_hat2 = self.ae2.decoder(z_gy)

        # IGAE decoding
        z_hat1, z_adj_hat1 = self.gae1.decoder(z_gx, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z_gy, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        return x_hat1, z_hat1, a_hat1, x_hat2, z_hat2, a_hat2, z_gx, z_gy, zg, cl_loss, Final_x_zinb, Final_x_ber
