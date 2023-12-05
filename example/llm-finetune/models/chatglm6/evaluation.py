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

import typing as t

from starwhale import handler, evaluation
from starwhale.api.service import api, LLMChat

try:
    from utils import BASE_MODEL_DIR, load_model_and_tokenizer
except ImportError:
    from .utils import BASE_MODEL_DIR, load_model_and_tokenizer


ds_input_keys = {
    "webqsp": "rawquestion",
    "grailqav1": "prompt",
    "graph_questions_testing": "prompt",
    "z_bench_common": "prompt",
    "mkqa": "query",
}

_g_model = None
_g_tokenizer = None


def get_model():
    global _g_tokenizer, _g_model
    if _g_tokenizer is None or _g_model is None:
        _g_model, _g_tokenizer = load_model_and_tokenizer()
        _g_model.eval()
    return _g_model, _g_tokenizer


@evaluation.predict(
    log_mode="plain",
    log_dataset_features=["query", "text", "question", "rawquestion", "prompt"],
    replicas=1,
)
def predict(data: dict, external: dict):
    chatglm, tokenizer = get_model()

    ds_name = external["dataset_uri"].name
    if ds_name in ds_input_keys:
        text = data[ds_input_keys[ds_name]]
    elif "text" in data:
        text = data["text"]
    elif "question" in data:
        text = data["question"]
    elif "rawquestion" in data:
        text = data["rawquestion"]
    elif "prompt" in data:
        text = data["prompt"]
    elif "query" in data:
        text = data["query"]
    else:
        raise ValueError(f"dataset {ds_name} does not fit this model")

    response, h = chatglm.chat(tokenizer, text, history=[])
    print(f"dataset: {text}\n chatglm6b: {response} \n")
    return response


@api(inference_type=LLMChat())
def online_eval(
    user_input: str,
    history: t.List[LLMChat.Message],
    top_p: float = 0.1,
    temperature: float = 0.5,
    max_new_tokens: int = 64,
) -> str:
    chatglm, tokenizer = get_model()
    chat_history = list()
    index = 0
    count = len(history)
    while index < count:
        if not history[index].bot:
            chat_history.append((history[index].content, history[index+1].content))
            index += 2
        else:
            raise ValueError("history is invalid")

    response, h = chatglm.chat(
        tokenizer=tokenizer,
        query=user_input,
        history=chat_history,
        max_length=max_new_tokens,
        top_p=top_p,
        temperature=temperature
    )
    print(f"input: {user_input}\n response: {response} \n")
    return response


@handler(expose=7860)
def chatbot():
    import gradio as gr

    chatglm, tokenizer = get_model()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        gr.ClearButton([msg, chatbot])
        max_length = gr.Slider(
            0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True
        )
        top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
        temperature = gr.Slider(
            0, 1, value=0.95, step=0.01, label="Temperature", interactive=True
        )

        def respond(message, chat_history, mxl, tpp, tmp):
            response, history = chatglm.chat(
                tokenizer,
                message,
                chat_history[-5] if len(chat_history) > 5 else chat_history,
                max_length=mxl,
                top_p=tpp,
                temperature=tmp,
            )
            chat_history.append((message, response))
            return "", chat_history

        msg.submit(
            respond, [msg, chatbot, max_length, top_p, temperature], [msg, chatbot]
        )

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    chatbot()
