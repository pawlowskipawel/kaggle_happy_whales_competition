# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models.ipynb (unless otherwise specified).

__all__ = ['GeM', 'Backbone', 'ArcMarginProduct', 'HappyWhalesModel']

# Cell
import torch.nn.functional as F
import torch.nn as nn
import torch
import timm
import math

# Cell
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



# Cell
class Backbone(nn.Module):
    def __init__(self, name="efficientnet_b0", pretrained=True):
        super().__init__()

        self.net = timm.create_model(name, pretrained=pretrained)

        if "efficientnet" in name:
            self.embedding_dim = self.net.classifier.in_features
        else:
            raise AttributeError("Wrong backbone name!")

    def forward(self, x):
        return self.net.forward_features(x)

# Cell
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0, device="cuda"):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        #if(CONFIG['enable_amp_half_precision']==True):
        cosine = cosine.to(torch.float32)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output



# Cell
class HappyWhalesModel(nn.Module):
    def __init__(self, model_name, embedding_dim, num_classes):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.model_name = model_name

        self.backbone = Backbone(self.model_name, pretrained=True)
        self.global_pool = GeM()

        self.neck = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(self.backbone.embedding_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU()
        )

        self.head = ArcMarginProduct(512, num_classes)

    def forward(self, x, label, return_embeddings=False):
        x = self.backbone(x)

        x = self.global_pool(x)
        x = x.flatten(1)

        x = self.neck(x)

        logits = self.head(x, label=label)

        if return_embeddings:
            return {'logits': logits, 'embeddings': x}
        else:
            return {'logits': logits}
