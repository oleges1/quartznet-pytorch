# import albumentations as album
import torchaudio
import random
import numpy as np
import torch
import string
# import youtokentome as yttm
# from torch.utils import data


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            try:
              data = t(data)
            except TypeError:
              # audiomentation transform
              data['audio'] = t(data['audio'], sample_rate=data['sample_rate'])
        return data


class AudioSqueeze:
    def __call__(self, data):
        data['audio'] = data['audio'].squeeze(0)
        return data


class AddLengths:
    def __call__(self, data):
        data['input_lengths'] = data['audio'].shape[-1]
        data['target_lengths'] = data['text'].shape[0]
        return data


class BPEtexts:
    def __init__(self, bpe, dropout_prob=0):
        self.bpe = bpe
        self.dropout_prob = dropout_prob

    def __call__(self, data):
        data['text'] = torch.tensor(self.bpe.encode(data['text'], dropout_prob=self.dropout_prob))
        return data


class TextPreprocess:
    def __call__(self, data):
        data['text'] = data['text'].lower().strip().translate(str.maketrans('', '', string.punctuation))
        return data


class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, data):
        data['audio'] = np.array(data['audio'])
        return data

# on gpu:

class ToGpu:
    def __init__(self, device):
        self.device = device

    def __call__(self, data):
        data = {k: [torch.from_numpy(item).to(self.device) for item in v] for k, v in data.items()}
        return data

class Pad:
    def __call__(self, data):
        padded_batch = {}
        for k, v in data.items():
            if len(v[0].shape) < 2:
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence([item[None] for item in v])
            else:
                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(v)
        return padded_batch

class MelSpectrogram(torchaudio.transforms.MelSpectrogram):
    def forward(self, data):
        for i in range(len(data['audio'])):
            data['audio'][i] = super(MelSpectrogram, self).forward(data['audio'][i])
        return data


class MaskSpectrogram(object):
    """Masking a spectrogram aka SpecAugment."""

    def __init__(self, frequency_mask_max_percentage=0.3, time_mask_max_percentage=0.1, probability=1.0):
        self.frequency_mask_probability = frequency_mask_max_percentage
        self.time_mask_probability = time_mask_max_percentage
        self.probability = probability

    def __call__(self, data):
        for i in range(len(data['audio'])):
            if random.random() < self.probability:
                nu, tau = data['audio'][i].shape

                f = random.randint(0, int(self.frequency_mask_probability*nu))
                f0 = random.randint(0, nu - f)
                data['audio'][i, f0:f0 + f, :] = 0

                t = random.randint(0, int(self.time_mask_probability*tau))
                t0 = random.randint(0, tau - t)
                data['audio'][i, :, t0:t0 + t] = 0

        return data
