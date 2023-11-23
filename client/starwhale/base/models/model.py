from __future__ import annotations

import typing as t

from pydantic.root_model import RootModel

from starwhale.base.models.base import SwBaseModel
from starwhale.base.client.models.models import ModelVo, StepSpec


class StepSpecClient(StepSpec):
    concurrency: t.Optional[int] = 1  # concurrency is deprecated in the sdk side
    replicas: int = 1
    func_name: str
    module_name: str
    cls_name: t.Optional[str] = None
    extra_args: t.Optional[t.List] = None
    extra_kwargs: t.Optional[t.Dict[str, t.Any]] = None


JobHandlers = RootModel[t.Dict[str, t.List[StepSpecClient]]]


class File(SwBaseModel):
    arcname: t.Optional[str] = None
    desc: str
    name: str
    path: str
    signature: str
    size: t.Optional[int] = None
    duplicate_check: bool


Files = RootModel[t.List[File]]


class LocalModelInfoBase(SwBaseModel):
    name: str
    version: str
    project: str
    path: str
    tags: t.Optional[t.List[str]] = None
    created_at: str
    is_removed: bool
    size: int


class LocalModelInfo(LocalModelInfoBase):
    handlers: t.Dict[str, t.List[StepSpecClient]]
    model_yaml: str
    files: t.Optional[t.List[File]] = None


ModelListType = t.Union[t.List[LocalModelInfoBase], t.List[ModelVo]]
