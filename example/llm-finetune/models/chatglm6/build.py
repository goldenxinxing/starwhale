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

import os

import starwhale
try:
    from utils import BASE_MODEL_DIR
    from finetune import fine_tune
    from evaluation import predict, chatbot, online_eval
except ImportError:
    from .utils import BASE_MODEL_DIR
    from .finetune import fine_tune
    from .evaluation import predict, chatbot, online_eval

starwhale.init_logger(3)


def build_starwhale_model() -> None:
    if not os.path.exists(BASE_MODEL_DIR):
        import download_model  # noqa: F401

    starwhale.model.build(
        name="chatglm6b",
        modules=[predict, chatbot, online_eval, fine_tune],
    )


if __name__ == "__main__":
    build_starwhale_model()
