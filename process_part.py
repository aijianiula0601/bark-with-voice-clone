import torch
import torch.nn as nn
import os
import re
import gc
import sys
import json
import math
import hashlib
import numpy as np
import logging
import torchaudio
from tqdm.auto import tqdm
import torch.nn.functional as F
from encodec.utils import convert_audio
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from packaging import version
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Pool
from functools import partial


max_duration_sec = 15.12 # the maximum allowed duration in seconds
hubert_tokenizer_path = 'data/models/hubert/tokenizer.pth'
hubert_path = 'data/models/hubert/hubert.pt'
device='cuda'

# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
from hubert.hubert_manager import HuBERTManager
hubert_manager = HuBERTManager()
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path=hubert_path).to(device)
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False

# Load the CustomTokenizer model
hubert_tokenizer = CustomTokenizer.load_from_checkpoint(hubert_tokenizer_path).to(device)  # Automatically uses the right layers

from bark.generation import load_codec_model
codec_model = load_codec_model(use_gpu=True)
codec_model.eval()
for param in codec_model.parameters():
    param.requires_grad = False


def get_duration(wav, sr):
    return wav.shape[1] / sr

print('loading done!')



CONTEXT_WINDOW_SIZE = 1024

MAX_SEMANTIC_LEN = 256

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599

MAX_COARSE_LEN = 768

SAMPLE_RATE = 24_000
CHANNELS = 1

COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

# 格式为： wav_path|text
path=sys.argv[1]#文件夹
train_path=sys.argv[2]#文件名

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8', errors='ignore') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        base = os.path.dirname(filename)
        for j in range(len(filepaths_and_text)):
            filepaths_and_text[j][0] = os.path.join(base, filepaths_and_text[j][0])
    return filepaths_and_text
    
valid_lines_train = []
# convert wavs to semantic tokens

for wav_path, txt in tqdm(load_filepaths_and_text(train_path)):
    save_path=os.path.join(path, 'tokens', os.path.basename(wav_path).replace('.wav', '.npz'))
    if os.path.exists(save_path):
        continue
    wav, sr = torchaudio.load(wav_path)
    if not get_duration(wav, sr) > max_duration_sec:
        valid_lines_train.append((wav_path, txt))
    wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS).to(device)

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=SAMPLE_RATE)
    semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)

    # save semantic tokens
    os.makedirs(os.path.join(path, 'tokens'), exist_ok=True)
    semantic_tokens = semantic_tokens.cpu().numpy()
    
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()

    # save tokens
    np.savez_compressed(save_path, fine=codes, coarse=codes[:2, :], semantic=semantic_tokens)


print('done!')