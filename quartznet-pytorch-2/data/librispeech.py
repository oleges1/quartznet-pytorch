from torch.utils import data
import torchaudio
import os
from torch.utils.data import Subset

class LibriDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, transforms, *args, **kwargs):
        if kwargs.get('download', False):
            os.makedirs(kwargs['root'], exist_ok=True)
        super(LibriDataset, self).__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        audio, sample_rate, text, _, _, _ = super().__getitem__(idx)
        return self.transforms({'audio' : audio, 'text': text, 'sample_rate': sample_rate})

    def get_text(self, idx):
        fileid = self._walker[idx]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + self._ext_txt
        file_text = os.path.join(self._path, speaker_id, chapter_id, file_text)

        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + self._ext_audio

        # Load text
        with open(file_text) as ft:
            for line in ft:
                fileid_text, utterance = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)

        return self.transforms({'text' : utterance})['text']


def get_dataset(config, transforms=lambda x: x, part='train'):
    if part == 'train':
        dataset = LibriDataset(root=config.dataset.root, url=config.dataset.get('train_url', 'train-clean-100'), download=True, transforms=transforms)
        return dataset
    elif part == 'val':
        dataset = LibriDataset(root=config.dataset.root, url=config.dataset.get('val_url', 'dev-clean'), download=True, transforms=transforms)
        return dataset
    elif part == 'bpe':
        dataset = LibriDataset(root=config.dataset.root, url=config.dataset.get('train_url', 'train-clean-100'), download=True, transforms=transforms)
        indices = list(range(len(dataset)))
        return dataset, indices
    else:
        raise ValueError('Unknown')
