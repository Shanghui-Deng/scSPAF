_author_ = 'lrren'
# coding: utf-8

_author_ = 'lrren'
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class DAE_Ber(nn.Module):

    def __init__(self, in_dim, hidden1, hidden2,  z_emb_size, dropout_rate):
        super(DAE_Ber, self).__init__()

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
        self.sig = nn.Sigmoid()
    def forward(self, x, scale_factor = 1.0 ):
        emb = self.fc_encoder(x)

        # expression matrix decoder
        latent = self.fc_decoder(emb)
        recon_x = self.decoder_scale(latent)
        Final_x = self.sig(recon_x)


        return emb, Final_x
