# quartznet-pytorch
Quartznet implementation on pytorch [https://arxiv.org/abs/1910.10261]

## Features:
 - Youtokentome tokenization with BPE dropout
 - Augmentations: custom and audiomentations
 - 3 datasets support: CommonVoice, Librispeech and LJSpeech
 - Weights & Biases logging
 - CTC beam search interation
 - GPU-based MelSpectrogram

## Trained models:



dataset | wer using dummy decoder | wer with ctc beam search | wer finetuned
--- | --- | --- | ---
LJspeech | [36.66](https://www.dropbox.com/s/9zn1rukf1pgunva/model_36_0.36604105617182675.pth?dl=0) | 34.45 | -
