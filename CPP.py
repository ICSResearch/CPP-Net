import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MSCA(nn.Module):
    def __init__(self, dim, dim2=None, reduction=8):
        super().__init__()
        self.dim = dim
        self.dim2 = dim2 if dim2 != None else self.dim
        self.pool_xw = nn.AdaptiveAvgPool2d((1, None))
        self.pool_xh = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_yw = nn.AdaptiveAvgPool2d((1, None))
        self.pool_yh = nn.AdaptiveAvgPool2d((None, 1))
        self.q = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim2 // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.dim2 // reduction, self.dim2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.k = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim2 // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.dim2 // reduction, self.dim2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.v = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim2, kernel_size=3, padding=1, groups=self.dim2),
            nn.GELU(),
            nn.Conv2d(self.dim2, self.dim2, kernel_size=1),
        )

    def forward(self, x, y):
        xw = self.pool_xw(x)  # b, c, 1, h
        xh = self.pool_xh(x)  # b, c, w, 1

        yw = self.pool_yw(y).reshape(xw.shape)  # b, c, 1, h
        yh = self.pool_yh(y).reshape(xh.shape)  # b, c, w, 1

        f1 = torch.matmul(xh, yw)
        f1 = self.q(f1)  # b, c, h, w

        f2 = torch.matmul(yh, xw)
        f2 = self.k(f2)  # b, c, h, w

        xt = self.v(x)
        xt = xt * f1 * f2
        return xt


class MSTB(nn.Module):
    def __init__(self, reduction, dim, dim2=None):
        super().__init__()
        self.cat = MSCA(dim=dim, dim2=dim2, reduction=reduction)
        dim3 = dim2 if dim2 != None else dim
        self.ln1 = LayerNorm(dim3, eps=1e-6, data_format="channels_first")
        self.ln3 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.ff = nn.Sequential(
            DCAB(dim3, reduction),
        )
        self.ln2 = LayerNorm(dim3, eps=1e-6, data_format="channels_first")

    def forward(self, x, y):
        x = self.ln1(x)
        y = self.ln3(y)
        x = self.cat(x, y) + x
        x = self.ln2(x)
        x = self.ff(x)
        return x


class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y


class DCAB(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0),
            CA(dim, reduction),
        )

    def forward(self, x):
        x = self.block(x) + x
        return x


class IterationModule(nn.Module):
    def __init__(self, block_size, ratio, dim, reduction):
        super(IterationModule, self).__init__()
        self.block_size = block_size
        self.delta = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        self.step = nn.Parameter(torch.zeros(1, dim, 1, 1) + 1e-3, requires_grad=True)
        self.lamda = nn.Parameter(torch.zeros(1, dim, 1, 1) + 1e-0, requires_grad=True)

        self.pre1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )
        self.pre2 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )

        self.down1 = nn.Sequential(
            DCAB(dim, reduction),
            nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down2 = nn.Sequential(
            DCAB(2 * dim, reduction),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size=2, stride=2),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down3 = nn.Sequential(
            DCAB(4 * dim, reduction),
            nn.Conv2d(4 * dim, 8 * dim, kernel_size=2, stride=2),
            LayerNorm(8 * dim, eps=1e-6, data_format="channels_first"),
        )

        self.mid = nn.Sequential(
            DCAB(8 * dim, reduction),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8 * dim, out_channels=4 * dim, kernel_size=2, stride=2
            ),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
            DCAB(4 * dim, reduction),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4 * dim, out_channels=2 * dim, kernel_size=2, stride=2
            ),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
            DCAB(2 * dim, reduction),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2 * dim, out_channels=dim, kernel_size=2, stride=2
            ),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            DCAB(dim, reduction),
        )

        self.unitc4 = MSTB(dim=8 * dim, dim2=dim, reduction=reduction)
        self.unitc5 = MSTB(dim=4 * dim, dim2=dim, reduction=reduction)
        self.unitc6 = MSTB(dim=2 * dim, dim2=dim, reduction=reduction)

        self.post1 = nn.Sequential(
            DCAB(3 * dim, reduction),
            nn.Conv2d(3 * dim, dim, kernel_size=1, padding=0),
            DCAB(dim, reduction),
        )
        self.post2 = nn.Sequential(
            DCAB(dim, reduction),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )

        self.aap = nn.AdaptiveAvgPool2d(1)
        self.thresh = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.i = 0

    def stres(self, x, mu):
        x_a = self.aap(x) * mu
        thr = self.thresh(x_a) * x_a
        z = torch.zeros_like(x)
        x = torch.sign(x) * torch.maximum(torch.abs(x) - thr, z) + x
        return x

    def forward(self, A, xk, yk, b, x1, x2, x3, x0, i):
        ############################## xk ##############################
        xp = xk

        ############################# DPFB #############################
        xk = self.pre1(xk)
        Aty = F.conv_transpose2d(yk, A, stride=self.block_size)
        Aty = self.pre2(Aty)
        xk = xk - self.step * Aty

        xk1 = self.down1(xk) + x1
        xk2 = self.down2(xk1) + x2
        xk3 = self.down3(xk2) + x3

        xk = self.stres(xk, self.step * self.lamda)

        xk4 = self.mid(xk3)
        xk_t = self.unitc4(xk, xk4)

        xk5 = self.up1(xk4) + xk2
        xk_t = self.unitc5(xk_t, xk5)

        xk6 = self.up2(xk5) + xk1
        xk_t = self.unitc6(xk_t, xk6)

        xk7 = self.up3(xk6) + xk

        xks = torch.cat([x0, xk_t, xk7], dim=1)
        x0 = self.post1(xks)
        xk = self.post2(x0)

        ############################## vk ##############################
        xkd = xk - xp + xk

        ############################## yk ##############################
        Axd = F.conv2d(xkd, A, stride=self.block_size, padding=0, bias=None)
        yk = (yk + self.delta * (Axd - b)) / (1 + self.delta)

        return xk, yk, xk6, xk5, xk4, x0


class CPP(torch.nn.Module):
    def __init__(self, ratio, block_size=32, dim=32, depth=8, reduction=8):
        super(CPP, self).__init__()
        self.ratio = ratio
        self.depth = depth
        self.block_size = block_size

        A = torch.from_numpy(self.load_sampling_matrix()).float()
        self.A = nn.Parameter(
            Rearrange("m (1 b1 b2) -> m 1 b1 b2", b1=self.block_size)(A),
            requires_grad=True,
        )

        self.pre = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            DCAB(dim, reduction),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down2 = nn.Sequential(
            DCAB(2 * dim, reduction),
            nn.Conv2d(2 * dim, dim * 4, kernel_size=2, stride=2),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down3 = nn.Sequential(
            DCAB(4 * dim, reduction),
            nn.Conv2d(4 * dim, dim * 8, kernel_size=2, stride=2),
            LayerNorm(8 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.iters = nn.ModuleList()
        for _ in range(self.depth):
            self.iters.append(IterationModule(self.block_size, ratio, dim, reduction))

        out_dim = (self.depth + 1) * (dim + 1)
        self.post = nn.Sequential(
            DCAB(out_dim, reduction),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            DCAB(dim, reduction),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        self.apply(self._init_weights)

    def forward(self, x):
        # Sampling
        b = F.conv2d(x, self.A, stride=self.block_size, padding=0, bias=None)

        # Init
        x_init = F.conv_transpose2d(b, self.A, stride=self.block_size)
        xk = x_init
        Ax = F.conv2d(x_init, self.A, stride=self.block_size, padding=0, bias=None)
        yk = Ax - b

        x0 = self.pre(xk)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # Recon
        xks = []
        xks.append(xk)
        xks.append(x0)
        for i in range(self.depth):
            xk, yk, x1, x2, x3, x0 = self.iters[i](self.A, xk, yk, b, x1, x2, x3, x0, i)
            xks.append(xk)
            xks.append(x0)

        xk = torch.cat(xks, dim=1)
        xk = self.post(xk)
        return xk

    def load_sampling_matrix(self):
        path = "./data/sampling_matrix"
        data = np.load(f"{path}/{self.ratio}_{self.block_size}.npy")
        return data

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_pretrained_model(ratio, model_name):
    path = f"./model/checkpoint-{model_name}-{ratio}-best.pth"
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


@register_model
def cpp9(ratio, pretrained=False, **kwargs):
    model = CPP(ratio, block_size=32, dim=32, depth=9, reduction=8)
    if pretrained:
        checkpoint = load_pretrained_model(ratio, sys._getframe().f_code.co_name)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def cpp8(ratio, pretrained=False, **kwargs):
    model = CPP(ratio, block_size=32, dim=32, depth=8, reduction=8)
    if pretrained:
        checkpoint = load_pretrained_model(ratio, sys._getframe().f_code.co_name)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def cpp7(ratio, pretrained=False, **kwargs):
    model = CPP(ratio, block_size=32, dim=32, depth=7, reduction=8)
    if pretrained:
        checkpoint = load_pretrained_model(ratio, sys._getframe().f_code.co_name)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def cpp5(ratio, pretrained=False, **kwargs):
    model = CPP(ratio, block_size=32, dim=32, depth=5, reduction=8)
    if pretrained:
        checkpoint = load_pretrained_model(ratio, sys._getframe().f_code.co_name)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def cpp3(ratio, pretrained=False, **kwargs):
    model = CPP(ratio, block_size=32, dim=32, depth=3, reduction=8)
    if pretrained:
        checkpoint = load_pretrained_model(ratio, sys._getframe().f_code.co_name)
        model.load_state_dict(checkpoint["model"])
    return model

