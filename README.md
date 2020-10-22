# quartznet-pytorch
Automatic Speech Recognition (ASR) on pytorch. Re-implementation on pytorch of Nvidia's [Quartznet](https://arxiv.org/abs/1910.10261).

## Features:
 - Youtokentome tokenization with BPE dropout
 - Augmentations: custom and audiomentations
 - 3 datasets support: CommonVoice, Librispeech and LJSpeech
 - Weights & Biases logging
 - CTC beam search interation
 - GPU-based MelSpectrogram

## Trained models:



dataset | wer using dummy decoder | wer with ctc beam search | wer finetuned dummy decoder | wer finetuned ctc beam search
--- | --- | --- | --- | ---
LJspeech | [36.66](https://www.dropbox.com/s/9zn1rukf1pgunva/model_36_0.36604105617182675.pth?dl=0) | 34.45 | [28.41](https://www.dropbox.com/s/5a9fxm5c2fxwh6x/model_17_0.2841321284154134.pth?dl=0) | 27.19

## W&B Logs:
 - [CommonVoiceRU](https://wandb.ai/oleges/quartznet_commonvoice_rus)
 - [LJSpeech](https://wandb.ai/oleges/quartznet_ljspeech)
