---
title: Hugging Face Transformer `THUDM/chatglm-6b`
---

```bash
$ python3 -m pip install starwhale
$ git clone https://github.com/star-whale/starwhale.git
$ cd starwhale/example/llm-finetune/models/chatglm6
$ python3 -m pip install -r requirements.txt
$ python3 download_model.py
$ swcli model build .
$ swcli dataset build -hf mkqa -n mkqa
```
