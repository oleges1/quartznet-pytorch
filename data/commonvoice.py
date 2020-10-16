from torch.utils import data
import torchaudio
import os
from torch.utils.data import Subset


# there is a bug with download dataset using torchudio code, maybe pull request

class CommonVoiceDataset(torchaudio.datasets.COMMONVOICE):
    def __init__(self, transforms, *args, **kwargs):
        if kwargs.get('download', False):
            os.makedirs(kwargs['root'], exist_ok=True)
        try:
            super(CommonVoiceDataset, self).__init__(*args, **kwargs)
        except FileNotFoundError:
            kwargs['root'] = '/'.join(kwargs['root'].split('/')[:-1])
            kwargs['version'] = ''
            kwargs['download'] = False
            super(CommonVoiceDataset, self).__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        audio, sample_rate, dic = super().__getitem__(idx)
        return self.transforms({'audio' : audio, 'text': dic['sentence'], 'sample_rate': sample_rate})

    def get_text(self, idx):
        line = self._walker[idx]
        return self.transforms({'text' : line[2]})['text']


def get_dataset(config, transforms=lambda x: x, part='train'):
    if part == 'train':
        dataset = CommonVoiceDataset(
            root=config.dataset.root,
            tsv=config.dataset.get('train_tsv', 'train.tsv'),
            url=config.dataset.get('language', 'english'),
            version=config.dataset.get('version', 'cv-corpus-4-2019-12-10'),
            download=config.dataset.get('download', True), transforms=transforms)
        return dataset
    elif part == 'val':
        dataset = CommonVoiceDataset(
            root=config.dataset.root,
            tsv=config.dataset.get('val_tsv', 'dev.tsv'),
            url=config.dataset.get('language', 'english'),
            version=config.dataset.get('version', 'cv-corpus-4-2019-12-10'),
            download=config.dataset.get('download', True), transforms=transforms)
        return dataset
    elif part == 'bpe':
        dataset = CommonVoiceDataset(
            root=config.dataset.root,
            tsv=config.dataset.get('train_tsv', 'train.tsv'),
            url=config.dataset.get('language', 'english'),
            version=config.dataset.get('version', 'cv-corpus-4-2019-12-10'),
            download=config.dataset.get('download', True), transforms=transforms)
        indices = list(range(len(dataset)))
        return dataset, indices
    else:
        raise ValueError('Unknown')
