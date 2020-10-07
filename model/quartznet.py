from torch import nn
import torch
from model.layers import MainBlock
from model.utils import init_weights

def QuartzNet(nn.Module):
    def __init__(
            self,
            model_config,
            feat_in,
            num_classes,
            activation='relu',
            normalization_mode="batch",
            norm_groups=-1,
            frame_splicing=1,
            init_mode='xavier_uniform',
            **kwargs
    ):
        super(QuartzNet, self).__init__()

        feat_in = feat_in * frame_splicing

        residual_panes = []
        layers = []
        for lcfg in model_config:
            dense_res = []
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            residual = lcfg.get('residual', True)
            layers.append(
                MainBlock(feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'] if 'dropout' in lcfg else 0.0,
                    residual=residual,
                    groups=groups,
                    separable=separable,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(layers)
        self.classify = nn.Conv1d(1024, num_classes,
                      kernel_size=1, bias=True)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal):
        feat = self.encoder(audio_signal)
        # BxCxT
        return self.classify(feat)

    def load_weights(self, path, map_location='cpu'):
        weights = torch.load(path, map_location=map_location)
        print(self.load_state_dict(weights, strict=False))
