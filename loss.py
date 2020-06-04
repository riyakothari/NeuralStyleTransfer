import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b,c,h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G

class StyleLoss(nn.Module):
    def forward(self, input, target):
        # gram_matrix = GramMatrix()
        # mseLoss = nn.MSELoss()
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)
