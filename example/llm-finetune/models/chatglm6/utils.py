#  Copyright 2022 Starwhale, Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import typing as t
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel, set_seed, AutoConfig

ROOT_DIR = Path(__file__).parent
BASE_MODEL_DIR = ROOT_DIR / "pretrain"
TUNED_MODEL_PATH = ROOT_DIR / "pretrain" / "pytorch_model.bin"


def load_model_and_tokenizer(quantization_bit=None, pre_seq_len=128, seed=0, prefix_projection=False) -> t.Tuple:
    # Set seed before initializing model.
    set_seed(seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        pre_seq_len=pre_seq_len,
        prefix_projection=prefix_projection,
    )

    # load tokenizer and model
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    _model = AutoModel.from_pretrained(BASE_MODEL_DIR, config=config, trust_remote_code=True)

    if TUNED_MODEL_PATH.exists():
        print(f"load p-tuning model: {TUNED_MODEL_PATH}")
        prefix_state_dict = torch.load(TUNED_MODEL_PATH)
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        _model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if quantization_bit is not None:
        print(f"Quantized to {quantization_bit} bit")
        _model = _model.quantize(quantization_bit)
    if pre_seq_len is not None:
        # P-tuning v2
        _model = _model.half()
        _model.transformer.prefix_encoder.float()
    else:
        # Finetune
        _model = _model.float()

    _model.gradient_checkpointing_enable()
    _model.enable_input_require_grads()

    return _model, _tokenizer
