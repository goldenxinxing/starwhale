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
import sys
import typing as t

from evaluation import predict, chatbot, online_eval

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from starwhale import dataset, Dataset
from starwhale.api import model, experiment

try:
    from utils import load_lora_config, BASE_MODEL_DIR, TUNED_MODEL_PATH, ROOT_DIR, load_model_and_tokenizer
except ImportError:
    from .utils import load_lora_config, BASE_MODEL_DIR, TUNED_MODEL_PATH, ROOT_DIR, load_model_and_tokenizer


ds_key_selectors = {
    "webqsp": {"rawquestion": "prompt", "parses[0].Answers[0].EntityName": "response"},
    "grailqav1": {"answer[0].entity_name": "response"},
    "graph_questions_testing": {"answer[0]": "response"},
    "z_bench_common": {"gpt4": "response"},
    "mkqa": {"query": "prompt", "answers.en[0].text": "response"},
}


# https://huggingface.co/THUDM/chatglm-6b/discussions/1
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
@experiment.finetune(resources={"nvidia.com/gpu": 1}, require_train_datasets=True, auto_build_model=True)
def p_tuning(
    dataset_uris: t.List[Dataset],
    ignore_pad_token_for_loss=False,
    quantization_bit=None,
    pre_seq_len=128
) -> None:
    tokenizer, chatglm = load_model_and_tokenizer(
        quantization_bit=quantization_bit,
        pre_seq_len=pre_seq_len,
    )

    sw_dataset = dataset_uris[0]
    sw_dataset = sw_dataset.with_loader_config(
        field_transformer=ds_key_selectors.get(sw_dataset._uri.name, None)
    )

    # Data collator
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=chatglm,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=chatglm,
        args=Seq2SeqTrainingArguments(
            "output",
            fp16=True,
            save_steps=500,
            save_total_limit=3,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            max_steps=1500,
            logging_steps=50,
            remove_unused_columns=False,
            seed=0,
            data_seed=0,
            group_by_length=False,
            dataloader_pin_memory=False,
        ),
        train_dataset=sw_dataset.to_pytorch(transform=SwDataTransform(
            tokenizer, ignore_pad_token_for_loss=ignore_pad_token_for_loss,
        )),
        tokenizer=tokenizer,
        data_collator=data_collator,
        save_prefixencoder=pre_seq_len is not None
    )

    trainer.train()

    model.build(
        workdir=ROOT_DIR,
        name="chatglm6b",
        modules=[predict, online_eval, chatbot, p_tuning],
    )


class SwDataTransform:
    def __init__(
        self,
        tokenizer,
        source_prefix="",
        prompt_column="prompt",
        response_column="response",
        history_column="history",
        max_source_length=200,
        max_target_length=500,
        ignore_pad_token_for_loss=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def __call__(self, features):
        return self.preprocess_function_train(features)

    def preprocess_function_train(self, example: t.Dict):
        query, answer = example[self.prompt_column], example[self.response_column]

        if self.history_column is None or example.get(self.history_column) is None:
            prompt = query
        else:
            prompt = ""
            history = example[self.history_column]
            for turn_idx, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

        prompt = self.source_prefix + prompt
        a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = self.tokenizer.encode(text=answer if answer else "", add_special_tokens=False)

        if len(a_ids) > self.max_source_length - 1:
            a_ids = a_ids[: self.max_source_length - 1]

        if len(b_ids) > self.max_target_length - 2:
            b_ids = b_ids[: self.max_target_length - 2]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

        context_length = input_ids.index(self.tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position + 1:]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        if self.ignore_pad_token_for_loss:
            labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


if __name__ == "__main__":
    ds_uri = sys.argv[0] or "cloud://e2e1113/project/starwhale/dataset/mkqa-train/version/euyjksjtobfm5kepxkq6t5ciztfju4oua725mnhh"
    p_tuning(dataset_uris=[dataset(ds_uri, readonly=True, create="forbid")])
