"""
Copyright (c) 2026 Genera1Z
https://github.com/Genera1Z
"""

from copy import deepcopy

from einops import rearrange, repeat
import torch as pt

from .randsfq import RandSFQ


class RandSFQ3(RandSFQ):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,
        decode,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            transit,
            decode,
        )
        self.transit_inv = deepcopy(self.transit)
        self.transit_inv.proji = self.transit.proji
        self.transit_inv.transit = self.transit.transit

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c0, h0, w0 = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        slotz = None
        attenta = []

        for i in range(t):
            if i == 0:  # (b,n,c)
                query_i = self.initializ(b if condit is None else condit[:, 0, :, :])
            else:  # slotz: [0,i); encode: [0,i]
                query_i = self.transit(slotz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_i, attenta_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )

            slotz = (  # (b,i+1,n,c)
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            attenta.append(attenta_i)  # t*(b,n,h*w)

        attenta = pt.stack(attenta, 1)  # (b,t,n,h*w)
        attenta = rearrange(attenta, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        recon, attentd = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)

        slotz_inv, attenta_inv, recon_inv, attentd_inv = self.aggregat_decode_inv(
            feature.detach(), encode, slotz  # > recon, recon.detach()
        )

        return (
            feature,
            slotz,
            attenta,
            recon,
            attentd,
            slotz_inv,
            attenta_inv,
            recon_inv,
            attentd_inv,
        )

    def aggregat_decode_inv(self, feature, encode, slotz):
        """
        - feature: shape=(b,t,c,h,w)
        - encode: shape=(b,t,hw,c)
        - slotz: shape=(b,t,s,c)
        """
        b, t, c, h, w = feature.shape
        slotz_inv = slotz[:, -1, :, :].detach()[:, None, :, :]
        attenta_inv = []

        for i in range(t - 2, -1, -1):
            # inverse, skip the last frame
            query_i_inv = self.transit_inv(slotz_inv, encode[:, i:, :, :].flip(1))
            slotz_i_inv, attenta_i_inv = self.aggregat(
                encode[:, i, :, :], query_i_inv, num_iter=1
            )
            slotz_inv = pt.concat([slotz_inv, slotz_i_inv[:, None, :, :]], 1)
            attenta_inv.append(attenta_i_inv)

        # assert slotz_inv.size(1) == 6
        # assert len(attent_inv) == 5
        slotz_inv = slotz_inv[:, 1:, :, :].flip(1)  # (b,t-1,s,c)
        attenta_inv = pt.stack(attenta_inv, 1)
        attenta_inv = rearrange(attenta_inv, "b t s (h w) -> b t s h w", h=h)

        clue = rearrange(feature[:, :-1, :, :, :], "b t c h w -> (b t) (h w) c")
        recon_inv, attentd_inv = self.decode(clue, slotz_inv.flatten(0, 1))
        recon_inv = rearrange(recon_inv, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd_inv = rearrange(attentd_inv, "(b t) s (h w) -> b t s h w", b=b, h=h)

        return slotz_inv, attenta_inv, recon_inv, attentd_inv
