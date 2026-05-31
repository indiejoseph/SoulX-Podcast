from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from soulxpodcast.models.modules.flow_components.estimator import \
    CausalConditionalDecoder
from soulxpodcast.models.modules.flow_components.upsample_encoder import (
    UpsampleConformerEncoder, make_pad_mask)


@dataclass
class CfmParams:
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7


class CausalConditionalCFM(torch.nn.Module):
    def __init__(self, in_channels=320, cfm_params=CfmParams(), n_spks=1, spk_emb_dim=80,
                 estimator: torch.nn.Module = None, meanflow: bool = False):
        super().__init__()
        self.n_feats = in_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # MeanFlow distillation produces a 1-step (or few-step) student that
        # predicts the average velocity over [t, r]. The estimator must be
        # built with the matching time-embedding fusion path; route the flag
        # through unless an estimator is supplied externally.
        self.meanflow = meanflow
        if estimator is None:
            self.estimator = CausalConditionalDecoder(meanflow=meanflow)
        else:
            self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None,
                streaming=False, meanflow: Optional[bool] = None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps. With meanflow, set
                n_timesteps=1 for the 1-step distilled inference path.
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            meanflow: per-call override of the init-time `self.meanflow` flag.
                When True (or when the model was built with meanflow=True), the
                cosine t-scheduler is bypassed and inference uses `basic_euler`
                (no CFG; distilled meanflow models already bake CFG in during
                training, per the upstream Chatterbox notes).

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if meanflow is None:
            meanflow = self.meanflow
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if (not meanflow) and self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        if meanflow:
            return self.basic_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks,
                                    cond=cond, streaming=streaming), None
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks,
                                cond=cond, streaming=streaming), None

    def basic_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """Single-batch Euler solver for MeanFlow inference.

        Each step calls the estimator once (no CFG batch doubling) with both
        the current time `t` and the step end-time `r`; the estimator returns
        the average velocity over [t, r]. For the distilled 1-step path use
        n_timesteps=1 — t_span becomes [0, 1] and the loop runs once.

        CFG is omitted: per the Chatterbox upstream, MeanFlow students are
        distilled from a teacher that was already CFG'd, so the guidance is
        baked into the weights.

        Args mirror solve_euler.
        """
        # Iterate consecutive pairs (t_i, t_{i+1}) along t_span.
        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(0)
            r = r.unsqueeze(0)
            dxdt = self.estimator(
                x, mask,
                mu, t,
                spks,
                cond,
                streaming,
                r,
            )
            dt = r - t
            x = x + dt * dxdt
        return x.float()

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        batch_size = x.size(0)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # Create tensors with double batch size for CFG (conditional + unconditional)
        x_in = torch.zeros([batch_size * 2, x.size(1), x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([batch_size * 2, mask.size(1), mask.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([batch_size * 2, mu.size(1), mu.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([batch_size * 2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([batch_size * 2, spks.size(1)], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([batch_size * 2, cond.size(1), cond.size(2)], device=x.device, dtype=x.dtype)

        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            # Copy conditional and unconditional input
            x_in[:batch_size] = x
            x_in[batch_size:] = x
            mask_in[:batch_size] = mask
            mask_in[batch_size:] = mask
            mu_in[:batch_size] = mu
            # Unconditional part remains 0
            t_in.fill_(t)
            spks_in[:batch_size] = spks
            cond_in[:batch_size] = cond

            dphi_dt = self.estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        meanflow: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = UpsampleConformerEncoder() if encoder is None else encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        # Same propagation rule as CausalConditionalCFM: if the caller supplies
        # a pre-built decoder it owns the flag; otherwise build one with the
        # right meanflow architecture so checkpoint shapes line up.
        self.meanflow = meanflow
        if decoder is None:
            self.decoder = CausalConditionalCFM(meanflow=meanflow)
        else:
            self.decoder = decoder
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def forward(self,
                token,
                token_len,
                prompt_feat,
                prompt_feat_len,
                embedding,
                streaming,
                finalize,
                n_timesteps: int = 15,
                meanflow: Optional[bool] = None):
        if n_timesteps <= 0:
            raise ValueError(f"n_timesteps must be positive, got {n_timesteps}")

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len, max_len=token.shape[1])).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(token, token_len, context=context, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros_like(h, device=token.device)
        for i, j in enumerate(prompt_feat_len):
            conds[i, :j] = prompt_feat[i, :j]
        conds = conds.transpose(1, 2)

        h_lengths = h_lengths.sum(dim=-1).squeeze(dim=1)
        mask = (~make_pad_mask(h_lengths, max_len=h.shape[1])).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            streaming=streaming,
            meanflow=meanflow,  # None defers to decoder's init-time flag
        )  # [B, num_mels, T]
        return feat.float(), h_lengths
