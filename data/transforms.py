# import albumentations as album
import torchaudio
import random
# from torch.utils import data

class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class AddLengths:
    def __call__(self, data):
        data['input_lengths'] = data['audio'].shape[1]
        data['target_lengths'] = data['text'].shape[0]
        return data

class AudioSqueeze:
    def __call__(self, data):
        data['audio'] = data['audio'].squeeze(0)
        return data

class BPEtexts:
    def __init__(self, bpe, dropout_prob=0):
        self.bpe = bpe
        self.dropout_prob = dropout_prob

    def __call__(self, data):
        data['text'] = torch.tensor(bpe.encode(data['text'], dropout_prob=self.dropout_prob))
        return data


class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, data):
        data['audio'] = np.array(data['audio'])
        return data


class MelSpectrogram(torchaudio.transforms.MelSpectrogram):
    def forward(self, data):
        data['audio'] = super(MelSpectrogram, self).forward(torch.tensor(data['audio']))
        return data


class MaskSpectrogram(object):
    """Masking a spectrogram aka SpecAugment."""

    def __init__(self, frequency_mask_max_percentage=0.3, time_mask_max_percentage=0.1, probability=1.0):
        self.frequency_mask_probability = frequency_mask_max_percentage
        self.time_mask_probability = time_mask_max_percentage
        self.probability = probability

    def __call__(self, data):
        if random.random() < self.probability:
            spectrogram = data['audio']
            nu, tau = spectrogram.shape

            f = random.randint(0, int(self.frequency_mask_probability*nu))
            f0 = random.randint(0, nu - f)
            spectrogram[f0:f0 + f, :] = 0

            t = random.randint(0, int(self.time_mask_probability*tau))
            t0 = random.randint(0, tau - t)
            spectrogram[:, t0:t0 + t] = 0

            data['audio'] = spectrogram

        return data
