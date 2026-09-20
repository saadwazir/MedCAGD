import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TIMM for universal encoders
from timm import create_model

# Configuration (contains decoder_width = 256)
from cfgs import *



def get_num_groups(num_channels):
    for g in [32, 16, 8, 4, 2, 1]:

        if num_channels % g == 0 and 4 <= (num_channels // g) <= 32:
            return g

    return 1





class ECABlock(nn.Module):
    def __init__(
        self,
        channels,
        gamma=2,
        b=1,
        pool_scales=(1, 2, 4),
        is_encoder=True,
        tau_init=1.0,
        tau_min=0.5,
        tau_max=2.0,
    ):
        super().__init__()


        self.channels = channels
        self.is_encoder = is_encoder


        if not is_encoder:
            gamma *= 1.2
            b *= 0.8


        self.k_size = self.get_kernel_size(channels, gamma, b)


        self.pool_scales = pool_scales

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.k_size,
                padding=(self.k_size - 1) // 2,
                bias=False
            )
            for _ in self.pool_scales
        ])


        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))
        self.tau_min = tau_min
        self.tau_max = tau_max


        self.sigmoid = nn.Sigmoid()


    def get_kernel_size(self, C, gamma, b):
        k = int(abs((math.log2(C) / gamma) + b))
        return k + 1 if k % 2 == 0 else k


    def forward(self, x):

        att_sum = 0


        for pool_size, conv in zip(self.pool_scales, self.convs):

            pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))


            pooled = pooled.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]


            logits = conv(
                pooled.squeeze(-1).transpose(-1, -2)
            ).transpose(-1, -2).unsqueeze(-1)


            att_sum += logits


        logits = att_sum / len(self.pool_scales)


        tau = torch.exp(self.log_tau)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)


        att = self.sigmoid(logits / tau)



        att = att / (att.mean(dim=1, keepdim=True) + 1e-6)


        return x * att






class NonLocalBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()


        self.channels = channels


        inter_channels = max(1, channels // reduction)


        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)


        self.transform = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.LayerNorm([inter_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False)
        )


    def forward(self, x):
        B, C, H, W = x.shape


        mask = self.conv_mask(x)                 # [B, 1, H, W]


        mask = mask.view(B, 1, H * W)


        attn = torch.softmax(mask, dim=-1)       # [B, 1, HW]


        x_flat = x.view(B, C, H * W)


        context = torch.bmm(
            x_flat, attn.transpose(1, 2)
        )                                         # [B, C, 1]


        context = context.view(B, C, 1, 1)


        out = self.transform(context)


        return out






class ResidualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()


        self.res_attn_block = NonLocalBlock(channels)


    def forward(self, x):

        attn_out = self.res_attn_block(x)


        return x + attn_out







class attention_gate(nn.Module):
    def __init__(self, g_c, x_c, inter_c):
        super().__init__()


        self.se_x = ECABlock(x_c, is_encoder=True)


        self.se_g = ECABlock(g_c, is_encoder=False)


        self.theta = nn.Sequential(
            nn.Conv2d(g_c, inter_c, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(inter_c), inter_c),
            nn.SiLU(),

            nn.Conv2d(
                inter_c, inter_c,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=inter_c,
                bias=False
            ),
            nn.GroupNorm(get_num_groups(inter_c), inter_c),
            nn.SiLU()
        )


        self.phi = nn.Sequential(
            nn.Conv2d(x_c, inter_c, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(inter_c), inter_c),
            nn.SiLU(),

            nn.Conv2d(
                inter_c, inter_c,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=inter_c,
                bias=False
            ),
            nn.GroupNorm(get_num_groups(inter_c), inter_c),
            nn.SiLU()
        )


        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(g_c + x_c, inter_c, 1),
            nn.SiLU(),
            nn.Sigmoid()
        )


        self.dw3 = nn.Conv2d(
            inter_c, inter_c,
            kernel_size=3,
            padding=1,
            groups=inter_c,
            bias=False
        )

        self.dw5 = nn.Conv2d(
            inter_c, inter_c,
            kernel_size=5,
            padding=2,
            groups=inter_c,
            bias=False
        )


        self.score_proj = nn.Conv2d(inter_c, 1, 1, bias=False)


        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.7)))
        self.tau_min = 0.3
        self.tau_max = 2.0


        self.att = nn.Sequential(
            nn.GroupNorm(1, inter_c),
            nn.Conv2d(inter_c, 1, 1),
            nn.Hardsigmoid()
        )


    def forward(self, g, x):


        x = self.se_x(x)
        g = self.se_g(g)


        theta_x = self.theta(g)
        phi_g = self.phi(x)


        fused = F.normalize(theta_x * phi_g, p=2, dim=1)


        fused = fused + self.global_fc(
            torch.cat([g, x], dim=1)
        ) * fused


        selector_feat = self.dw3(fused) + self.dw5(fused)
        score = self.score_proj(selector_feat)  # [B, 1, H, W]


        tau = torch.exp(self.log_tau)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        B, _, H, W = score.shape
        alpha_spatial = F.softmax(
            score.view(B, -1) / tau,
            dim=-1
        ).view(B, 1, H, W)


        fused = fused * (1.0 + alpha_spatial)


        alpha = self.att(fused + 1e-5)


        return x * alpha






class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()


        groups = get_num_groups(out_c)


        self.conv = nn.Sequential(


            nn.Conv2d(
                in_c, in_c,
                kernel_size=3,
                padding=1,
                groups=in_c,
                bias=False
            ),
            nn.GroupNorm(get_num_groups(in_c), in_c),
            nn.SiLU(),


            nn.Conv2d(
                in_c, out_c,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(groups, out_c),
            nn.SiLU(),


            nn.Conv2d(
                out_c, out_c,
                kernel_size=3,
                padding=1,
                groups=out_c,
                bias=False
            ),
            nn.GroupNorm(groups, out_c),
            nn.SiLU(),


            nn.Conv2d(
                out_c, out_c,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(groups, out_c),
            nn.SiLU(),
        )


    def forward(self, x):
        return self.conv(x)



class decoder_block(nn.Module):
    def __init__(self, g_c, x_c, out_c):
        super().__init__()


        self.up = nn.ConvTranspose2d(
            g_c, out_c,
            kernel_size=2,
            stride=2
        )


        self.use_att = (x_c > 64)


        if x_c > 0:

            if self.use_att:

                self.att = attention_gate(
                    out_c,           # decoder channels (guidance)
                    x_c,             # encoder channels (to be gated)
                    max(1, x_c // 2) # intermediate latent width
                )
            else:

                self.att = None


            self.conv = conv_block(
                out_c + x_c,
                out_c
            )
        else:

            self.att = None
            self.conv = conv_block(out_c, out_c)


    def forward(self, g, x):


        g = self.up(g)


        if x is not None:


            if g.shape[2:] != x.shape[2:]:
                g = F.interpolate(
                    g,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            if self.att is not None:
                x = self.att(g, x)


            g = torch.cat([g, x], dim=1)


        return self.conv(g)



class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch=512):
        super().__init__()


        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)


        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.SiLU(),

            # Injects global spatial context
            ResidualAttention(out_ch)
        )


    def forward(self, x):
        return self.refine(self.proj(x))



class RefinementBlock(nn.Module):
    def __init__(self, ch, spatial_focus=False):
        super().__init__()


        self.conv = conv_block(ch, ch)


        self.att = ECABlock(ch)


        self.spatial = (
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
            if spatial_focus else nn.Identity()
        )


    def forward(self, x):

        return self.att(self.spatial(self.conv(x)))


class ContextAggregator(nn.Module):
    def __init__(self, in_channels, out_ch):
        super().__init__()


        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_ch, kernel_size=1)
            for c in in_channels
        ])


        self.global_attn = ResidualAttention(out_ch)


    def forward(self, feats, target_size):


        fused = 0
        for f, proj in zip(feats, self.proj):
            f = F.interpolate(
                proj(f),
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
            fused = fused + f


        return self.global_attn(fused / len(feats))



class build_model(nn.Module):
    def __init__(self, encoder_name=encoder_name_str):
        super().__init__()


        self.encoder = create_model(
            encoder_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )


        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            feats = self.encoder(dummy)


            if feats[0].shape[1] < feats[0].shape[-1]:
                feats = [f.permute(0, 3, 1, 2) for f in feats]


            real_channels = [f.shape[1] for f in feats]


        DEC_C = [64, 128, 320, 512]  # p1, p2, p3, p4


        self.p1 = nn.Conv2d(real_channels[0], DEC_C[0], 1)
        self.p2 = nn.Conv2d(real_channels[1], DEC_C[1], 1)
        self.p3 = nn.Conv2d(real_channels[2], DEC_C[2], 1)
        self.p4 = nn.Conv2d(real_channels[3], DEC_C[3], 1)


        self.bottleneck = Bottleneck(DEC_C[3], DEC_C[3])


        self.d1 = decoder_block(512, 320, 320)
        self.d2 = decoder_block(320, 128, 128)
        self.d3 = decoder_block(128, 64, 64)
        self.d4 = decoder_block(64, 64, 64)


        self.ref1 = RefinementBlock(320, spatial_focus=False)
        self.ref2 = RefinementBlock(128, spatial_focus=True)
        self.ref3 = RefinementBlock(64, spatial_focus=True)
        self.ref4 = RefinementBlock(64, spatial_focus=True)


        self.context1 = ContextAggregator([64, 128, 320, 512], 320)
        self.context2 = ContextAggregator([64, 128, 320, 512], 128)
        self.context3 = ContextAggregator([64, 128, 320, 512], 64)
        self.context4 = None  # full-resolution context omitted


        self.skip0_proj = nn.Conv2d(3 + DEC_C[0], DEC_C[0], 1)


        out_ch = num_Classes if num_Classes > 2 else 1

        self.final    = nn.Conv2d(64, out_ch, 1)
        self.final_d1 = nn.Conv2d(320, out_ch, 1)
        self.final_d2 = nn.Conv2d(128, out_ch, 1)
        self.final_d3 = nn.Conv2d(64, out_ch, 1)


        self.edge_d1 = nn.Conv2d(320, 1, 1)
        self.edge_d2 = nn.Conv2d(128, 1, 1)
        self.edge_d3 = nn.Conv2d(64, 1, 1)


    def forward(self, x):


        feats = self.encoder(x)
        c1, c2, c3, c4 = feats


        if c1.shape[1] < c1.shape[-1]:
            c1 = c1.permute(0, 3, 1, 2)
            c2 = c2.permute(0, 3, 1, 2)
            c3 = c3.permute(0, 3, 1, 2)
            c4 = c4.permute(0, 3, 1, 2)


        p1 = self.p1(c1)
        p2 = self.p2(c2)
        p3 = self.p3(c3)
        p4 = self.p4(c4)


        b = self.bottleneck(p4)


        d1 = self.d1(b, p3)
        d1 = self.ref1(d1 + self.context1([p1, p2, p3, b], d1.shape[2:]))

        d2 = self.d2(d1, p2)
        d2 = self.ref2(d2 + self.context2([p1, p2, p3, b], d2.shape[2:]))

        d3 = self.d3(d2, p1)
        d3 = self.ref3(d3 + self.context3([p1, p2, p3, b], d3.shape[2:]))


        p1_up = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=False)
        skip0 = self.skip0_proj(torch.cat([x, p1_up], dim=1))

        d4 = self.d4(d3, skip0)
        d4 = self.ref4(d4)


        out_main = self.final(d4)
        out_main = F.interpolate(out_main, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Deep supervision outputs
        aux1 = F.interpolate(self.final_d1(d1), size=x.shape[2:], mode='bilinear', align_corners=False)
        aux2 = F.interpolate(self.final_d2(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        aux3 = F.interpolate(self.final_d3(d3), size=x.shape[2:], mode='bilinear', align_corners=False)

        # Edge supervision outputs
        edge1 = F.interpolate(self.edge_d1(d1), size=x.shape[2:], mode='bilinear', align_corners=False)
        edge2 = F.interpolate(self.edge_d2(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        edge3 = F.interpolate(self.edge_d3(d3), size=x.shape[2:], mode='bilinear', align_corners=False)


        return out_main, [aux1, aux2, aux3], [edge1, edge2, edge3]

