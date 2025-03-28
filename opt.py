import argparse

parser = argparse.ArgumentParser(description='scSPAF', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="PBMC-10k")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--rec_epoch', type=int, default=30)
parser.add_argument('--fus_epoch', type=int, default=300)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--pretrain', type=bool, default=False)

# parameters
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--alpha_value', type=float, default=0.1)
parser.add_argument('--lambda1', type=float, default=10)
parser.add_argument('--lambda2', type=float, default=0.1)
parser.add_argument('--lambda3', type=float, default=10)
parser.add_argument('--method', type=str, default='euc')
parser.add_argument('--first_view', type=str, default='ATAC')
parser.add_argument('--lr', type=float, default=7e-04)

# dimension of input and latent representations
parser.add_argument('--n_d1', type=int, default=100)
parser.add_argument('--n_d2', type=int, default=100)
parser.add_argument('--n_z', type=int, default=20)

# AE structure parameter
parser.add_argument('--ae_n_enc_1', type=int, default=256)
parser.add_argument('--ae_n_enc_2', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=256)

# IGAE structure parameter
parser.add_argument('--gae_n_enc_1', type=int, default=256)
parser.add_argument('--gae_n_enc_2', type=int, default=128)
parser.add_argument('--gae_n_dec_1', type=int, default=128)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--ami', type=float, default=0)


parser.add_argument('--acc2', type=float, default=0)
parser.add_argument('--nmi2', type=float, default=0)
parser.add_argument('--ari2', type=float, default=0)
parser.add_argument('--ami2', type=float, default=0)

parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)

args = parser.parse_args()
