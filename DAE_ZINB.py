_author_ = 'lrren'
# coding: utf-8

_author_ = 'lrren'
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAE_ZINB(nn.Module):

    def __init__(self, in_dim, hidden1, hidden2,  z_emb_size, dropout_rate):
        super(DAE_ZINB, self).__init__()

        # self.params = args
        self.in_dim = in_dim

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),


            nn.Linear(hidden2, z_emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Distribution reconstruction
        self.fc_decoder = nn.Sequential(
            nn.Linear(z_emb_size, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        self.decoder_scale = nn.Linear(hidden1, self.in_dim)
        self.decoder_r = nn.Linear(hidden1, self.in_dim)
        self.dropout = nn.Linear(hidden1, self.in_dim)

    def forward(self, x, scale_factor = 1.0 ):
        emb = self.fc_encoder(x)

        # expression matrix decoder
        latent = self.fc_decoder(emb)

        normalized_x = F.softmax(self.decoder_scale(latent), dim=1)

        batch_size = normalized_x.size(0)
        scale_factor.resize_(batch_size, 1)
        scale_factor.repeat(1, normalized_x.size(1))

        scale_x = torch.exp(scale_factor) * normalized_x  # recon_x
        # scale_x = normalized_x  # recon_x

        disper_x = torch.exp(self.decoder_r(latent))  ### theta
        dropout_rate = self.dropout(latent)

        return emb, normalized_x, dropout_rate, disper_x, scale_x