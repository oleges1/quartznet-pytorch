from torch.utils import data
import torchaudio

class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, transforms, *args, **kwargs):
        if kwargs.get('download', False):
            os.path.makedirs(kwargs['root'])
        super(LJSpeechDataset, self).__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        audio, sample_rate, _, norm_text = super().__getitem__(idx)
        return self.transforms({'audio' : audio, 'text': text, 'sample_rate': sample_rate})

    def get_text(self, n):
        line = self._walker[n]
        fileid, transcript, normalized_transcript = line
        return normalized_transcript
