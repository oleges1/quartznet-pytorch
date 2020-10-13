import os
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
import math
import argparse

import data
from utils import fix_seeds, remove_from_dict, prepare_bpe
from data.collate import collate_fn, gpu_collate, no_pad_collate
from data.transforms import (
        Compose, AddLengths, AudioSqueeze, TextPreprocess,
        MaskSpectrogram, ToNumpy, BPEtexts, MelSpectrogram,
        ToGpu, Pad
)

from functools import partial

# model:
from model import configs as quartznet_configs
from model.quartznet import QuartzNet

# utils:
import yaml
from easydict import EasyDict as edict
from utils import fix_seeds, remove_from_dict, prepare_bpe
import wandb
from decoder import GreedyDecoder, BeamCTCDecoder


def evaluate(config):
    fix_seeds(seed=config.train.get('seed', 42))
    dataset_module = getattr(data, config.dataset.name)
    bpe = prepare_bpe(config)

    transforms_val = Compose([
            TextPreprocess(),
            ToNumpy(),
            BPEtexts(bpe=bpe),
            AudioSqueeze()
    ])

    batch_transforms_val = Compose([
            ToGpu('cuda' if torch.cuda.is_available() else 'cpu'),
            MelSpectrogram(
                sample_rate=config.dataset.get('sample_rate', 16000), # for LJspeech
                n_mels=config.model.feat_in
            ).to('cuda' if torch.cuda.is_available() else 'cpu'),
            AddLengths(),
            Pad()
    ])

    val_dataset = dataset_module.get_dataset(config, transforms=transforms_val, part='val')
    val_dataloader = DataLoader(val_dataset, num_workers=config.train.get('num_workers', 4),
                batch_size=1, collate_fn=no_pad_collate)

    model = QuartzNet(
        model_config=getattr(quartznet_configs, config.model.name, '_quartznet5x5_config'),
        **remove_from_dict(config.model, ['name'])
    )
    print(model)

    if config.train.get('from_checkpoint', None) is not None:
        model.load_weights(config.train.from_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    decoder = BeamCTCDecoder(bpe=bpe)

    model.eval()
    val_stats = defaultdict(list)
    for batch_idx, batch in enumerate(val_dataloader):
        batch = batch_transforms_val(batch)
        with torch.no_grad():
            logits = model(batch['audio'])
            output_length = torch.ceil(batch['input_lengths'].float() / model.stride).int()
            loss = criterion(logits.permute(2, 0, 1).log_softmax(dim=2), batch['text'], output_length, batch['target_lengths'])

        target_strings = decoder.convert_to_strings(batch['text'])
        decoded_output = decoder.decode(logits.permute(0, 2, 1).softmax(dim=2))
        wer = np.mean([decoder.wer(true, pred) for true, pred in zip(target_strings, decoded_output)])
        cer = np.mean([decoder.cer(true, pred) for true, pred in zip(target_strings, decoded_output)])
        val_stats['val_loss'].append(loss.item())
        val_stats['wer'].append(wer)
        val_stats['cer'].append(cer)
    for k, v in val_stats.items():
        val_stats[k] = np.mean(v)
    val_stats['val_samples'] = wandb.Table(columns=['gt_text', 'pred_text'], data=zip(target_strings, decoded_output))
    print(val_stats)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation model.')
    parser.add_argument('--config', default='configs/train_LJSpeech.yml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = edict(yaml.safe_load(stream))
    evaluate(config)
