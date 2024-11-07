import torch
import torch.nn as nn


class Monotone28(nn.Module):
    def __init__(self, z_dim=100, ngf=128, nc=1, dropout=0):
        super(Monotone28, self).__init__()
        self.convt1 = self.conv_trans_layers(z_dim, ngf * 4, 3, 1, 0)
        self.dropout1 = nn.Dropout2d(dropout)
        # (ngf*4), 3, 3
        self.convt2 = self.conv_trans_layers(ngf * 4, ngf * 2, 3, 2, 0)
        self.dropout2 = nn.Dropout2d(dropout)
        # (ngf*2), 7, 7
        self.convt3 = self.conv_trans_layers(ngf * 2, ngf, 4, 2, 1)
        self.dropout3 = nn.Dropout2d(dropout)
        # (ngf*1), 14, 14
        self.convt4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1), nn.Sigmoid())
        # (nc), 28, 28

    @staticmethod
    def conv_trans_layers(in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return net

    def forward(self, x: torch.Tensor, receiver_input=None, aux_input=None):
        x = x.unsqueeze(-1).unsqueeze(-1)
        out = self.convt1(x)
        out = self.dropout1(out)
        out = self.convt2(out)
        out = self.dropout2(out)
        out = self.convt3(out)
        out = self.dropout3(out)
        out = self.convt4(out)
        return out


class Color32(nn.Module):
    def __init__(self, z_dim=100, ngf=128, nc=3, dropout=0):
        super(Color32, self).__init__()
        self.convt1 = self.conv_trans_layers(z_dim, ngf * 4, 4, 1, 0)
        self.dropout1 = nn.Dropout2d(dropout)
        # (ngf*4), 4, 4
        self.convt2 = self.conv_trans_layers(ngf * 4, ngf * 2, 4, 2, 1)
        self.dropout2 = nn.Dropout2d(dropout)
        # (ngf*2), 8, 8
        self.convt3 = self.conv_trans_layers(ngf * 2, ngf, 4, 2, 1)
        self.dropout3 = nn.Dropout2d(dropout)
        # (ngf*1), 16, 16
        self.convt4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1), nn.Sigmoid())
        # (nc), 32, 32

    @staticmethod
    def conv_trans_layers(in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return net

    def forward(self, x: torch.Tensor, receiver_input=None, aux_input=None):
        x = x.unsqueeze(-1).unsqueeze(-1)
        out = self.convt1(x)
        out = self.dropout1(out)
        out = self.convt2(out)
        out = self.dropout2(out)
        out = self.convt3(out)
        out = self.dropout3(out)
        out = self.convt4(out)
        return out
