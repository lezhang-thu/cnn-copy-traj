import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#class ImgChLayerNorm(nn.Module):
#
#    def __init__(self, ch, eps=1e-03):
#        super(ImgChLayerNorm, self).__init__()
#        self.norm = torch.nn.LayerNorm(ch, eps=eps)
#
#    def forward(self, x):
#        x = x.permute(0, 2, 3, 1)
#        x = self.norm(x)
#        x = x.permute(0, 3, 1, 2)
#        return x
#
#
#class ReversibleConvEncoder(nn.Module):
#    """Encoder that stores metadata for exact reverse decoding"""
#
#    def __init__(
#        self,
#        input_shape,
#        depth=32,
#        act="SiLU",
#        norm=True,
#        kernel_size=3,  # Using your intended kernel size
#        stride=2,
#        minres=4,
#    ):
#        super(ReversibleConvEncoder, self).__init__()
#        act_fn = getattr(torch.nn, act)
#        h, w, input_ch = input_shape
#
#        # Store original parameters for decoder
#        self.input_shape = input_shape
#        self.depth = depth
#        self.act = act
#        self.norm = norm
#        self.kernel_size = kernel_size
#        self.stride = stride
#        self.minres = minres
#
#        # Calculate stages based on minimum dimension to ensure symmetry
#        min_dim = min(h, w)
#        self.stages = int(np.log2(min_dim) - np.log2(minres))
#
#        # Store layer configurations for exact reversal
#        self.layer_configs = []
#        self.spatial_dims = [(h, w)]  # Track spatial dimensions at each stage
#
#        in_dim = input_ch
#        out_dim = depth
#        current_h, current_w = h, w
#
#        layers = []
#        for i in range(self.stages):
#            # Calculate padding to ensure proper downsampling
#            pad_h = (kernel_size - stride) // 2 if (
#                kernel_size - stride) % 2 == 0 else (kernel_size - stride) // 2
#            pad_w = pad_h  # Symmetric padding
#
#            # Calculate actual output dimensions
#            new_h = (current_h + 2 * pad_h - kernel_size) // stride + 1
#            new_w = (current_w + 2 * pad_w - kernel_size) // stride + 1
#
#            # Store configuration for decoder reversal
#            config = {
#                'in_channels': in_dim,
#                'out_channels': out_dim,
#                'kernel_size': kernel_size,
#                'stride': stride,
#                'padding': (pad_h, pad_w),
#                'input_size': (current_h, current_w),
#                'output_size': (new_h, new_w),
#                'has_norm': norm,
#            }
#            self.layer_configs.append(config)
#
#            # Add conv layer
#            layers.append(
#                nn.Conv2d(
#                    in_channels=in_dim,
#                    out_channels=out_dim,
#                    kernel_size=kernel_size,
#                    stride=stride,
#                    padding=(pad_h, pad_w),
#                    bias=not norm,  # No bias if using norm
#                ))
#
#            # Add normalization
#            if norm:
#                layers.append(ImgChLayerNorm(out_dim))
#
#            # Add activation
#            layers.append(act_fn())
#
#            # Update for next iteration
#            in_dim = out_dim
#            out_dim *= 2
#            current_h, current_w = new_h, new_w
#            self.spatial_dims.append((current_h, current_w))
#
#            print(
#                f'Encoder Stage {i}: ({config["input_size"]}) -> ({config["output_size"]}) channels: {config["in_channels"]} -> {config["out_channels"]}'
#            )
#
#        # Store final output dimensions and channel count
#        self.output_shape = (current_h, current_w, in_dim)
#        self.outdim = in_dim * current_h * current_w
#
#        print(f'Final encoder output shape: {self.output_shape}')
#        print(f'Final encoder outdim: {self.outdim}')
#
#        self.layers = nn.Sequential(*layers)
#        # Apply weight initialization if tools available
#        try:
#            self.layers.apply(tools.weight_init)
#        except NameError:
#            pass  # tools not available
#
#    def forward(self, obs):
#        obs = obs - 0.5
#        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
#        x = obs.reshape((-1, ) + tuple(obs.shape[-3:]))
#        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
#        x = x.permute(0, 3, 1, 2)
#        x = self.layers(x)
#        # (batch * time, ...) -> (batch * time, -1)
#        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
#        return x


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        frame_stack,
        act_dim,
    ):
        super().__init__()

        #self.conv = ReversibleConvEncoder(
        #    input_shape=(96, 96, 4),
        #    depth=32,
        #    act="SiLU",
        #    norm=True,
        #    kernel_size=4,
        #    stride=2,
        #    minres=4,
        #)

        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.rnn = torch.nn.GRU(3136, 512)
        # build policy function
        self.pi = nn.Linear(512, act_dim)

    def forward(self, obs, h_0):
        obs = obs.to(torch.float32) / 255.0
        obs = obs.permute(0, 3, 1, 2)
        t, h_0 = self.rnn(self.conv(obs), h_0)
        logit = self.pi(t)
        return logit, h_0

    @torch.no_grad()
    def act(self, obs, deterministic, h_0):
        logit, h_0 = self(obs, h_0)
        if deterministic:
            actions = logit.argmax(1)
        else:
            actions = torch.distributions.categorical.Categorical(
                logits=logit, ).sample()
        return actions.squeeze(0).detach().cpu().numpy(), h_0
