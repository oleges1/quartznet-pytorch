from torch.utils import data
import torchaudio
import os

class LibriDataset(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, transforms, *args, **kwargs):
        if kwargs.get('download', False):
            os.makedirs(kwargs['root'])
        super(LibriDataset, self).__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        audio, sample_rate, text, _, _, _ = super().__getitem__(idx)
        return self.transforms({'audio' : audio, 'text': text, 'sample_rate': sample_rate})

    def get_text(self, idx):
        fileid = self._walker[idx]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + ext_txt
        file_text = os.path.join(path, speaker_id, chapter_id, file_text)

        # Load text
        with open(file_text) as ft:
            for line in ft:
                fileid_text, utterance = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)

        return utterance
