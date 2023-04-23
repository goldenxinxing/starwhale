import os
import typing as t
import pathlib
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner
from requests_mock import Mocker
from pyfakefs.fake_filesystem_unittest import TestCase

from tests import ROOT_DIR
from starwhale.utils import config as sw_config
from starwhale.utils import load_yaml
from starwhale.consts import (
    FileFlag,
    HTTPMethod,
    DefaultYAMLName,
    SW_AUTO_DIRNAME,
    VERSION_PREFIX_CNT,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_JOBS_FILE_NAME,
    EVALUATION_PANEL_LAYOUT_JSON_FILE_NAME,
    EVALUATION_PANEL_LAYOUT_YAML_FILE_NAME,
)
from starwhale.base.uri import URI
from starwhale.utils.fs import empty_dir, ensure_dir, ensure_file
from starwhale.base.type import URIType, BundleType
from starwhale.api.service import Service
from starwhale.utils.config import SWCliConfigMixed
from starwhale.api._impl.job import Handler
from starwhale.core.model.cli import _list as list_cli
from starwhale.core.model.view import ModelTermView, ModelTermViewJson
from starwhale.core.model.model import (
    ModelConfig,
    ModelInfoFilter,
    StandaloneModel,
    resource_to_file_node,
)
from starwhale.core.instance.view import InstanceTermView
from starwhale.base.scheduler.step import Step

_model_data_dir = f"{ROOT_DIR}/data/model"
_model_yaml = open(f"{_model_data_dir}/model.yaml").read()
_base_model_yaml = open(f"{_model_data_dir}/base_manifest.yaml").read()
_compare_model_yaml = open(f"{_model_data_dir}/compare_manifest.yaml").read()


class StandaloneModelTestCase(TestCase):
    def setUp(self) -> None:
        self.setUpPyfakefs()
        sw_config._config = {}

        self.sw = SWCliConfigMixed()

        self.workdir = "/home/starwhale/myproject"
        self.name = "mnist"

        self.fs.create_file(
            os.path.join(self.workdir, DefaultYAMLName.MODEL), contents=_model_yaml
        )

        ensure_dir(os.path.join(self.workdir, "models"))
        ensure_dir(os.path.join(self.workdir, "config"))
        ensure_file(os.path.join(self.workdir, "models", "mnist_cnn.pt"), " ")
        ensure_file(os.path.join(self.workdir, "config", "hyperparam.json"), " ")

    @patch("starwhale.base.blob.store.LocalFileStore.copy_dir")
    @patch("starwhale.api._impl.job.Handler._preload_registering_handlers")
    @patch("starwhale.core.model.model.file_stat")
    @patch("starwhale.core.model.model.StandaloneModel._get_service")
    @patch("starwhale.core.model.model.Walker.files")
    @patch("starwhale.core.model.model.blake2b_file")
    def test_build_workflow(
        self,
        m_blake_file: MagicMock,
        m_walker_files: MagicMock,
        m_get_service: MagicMock,
        m_stat: MagicMock,
        m_preload: MagicMock,
        m_copy_dir: MagicMock,
    ) -> None:
        Handler._registered_handlers["base"] = Handler(
            name="base",
            show_name="t1",
            func_name="test_func1",
            module_name="mock",
        )
        Handler._registered_handlers["depend"] = Handler(
            name="depend",
            show_name="t2",
            func_name="test_func2",
            module_name="mock",
            needs=["base"],
        )
        m_stat.return_value.st_size = 1
        m_blake_file.return_value = "123456"
        m_walker_files.return_value = []

        svc = MagicMock(spec=Service)
        svc.get_spec.return_value = {}
        svc.example_resources = []
        m_get_service.return_value = svc

        model_config = ModelConfig.create_by_yaml(
            Path(self.workdir) / DefaultYAMLName.MODEL
        )

        model_uri = URI(self.name, expected_type=URIType.MODEL)
        sm = StandaloneModel(model_uri)
        sm.build(workdir=Path(self.workdir), model_config=model_config)

        build_version = sm.uri.object.version

        bundle_path = (
            self.sw.rootdir
            / "self"
            / URIType.MODEL
            / self.name
            / build_version[:VERSION_PREFIX_CNT]
            / f"{build_version}{BundleType.MODEL}"
        )
        assert m_preload.call_count == 1

        assert bundle_path.exists()
        assert (bundle_path / "src").exists()
        job_yaml_path = bundle_path / "src" / SW_AUTO_DIRNAME / DEFAULT_JOBS_FILE_NAME

        assert job_yaml_path.exists()
        job_contents = load_yaml(job_yaml_path)
        assert job_contents == {
            "base": [
                {
                    "cls_name": "",
                    "concurrency": 1,
                    "extra_args": [],
                    "extra_kwargs": {},
                    "func_name": "test_func1",
                    "module_name": "mock",
                    "name": "base",
                    "needs": [],
                    "replicas": 1,
                    "resources": [],
                    "show_name": "t1",
                }
            ],
            "depend": [
                {
                    "cls_name": "",
                    "concurrency": 1,
                    "extra_args": [],
                    "extra_kwargs": {},
                    "func_name": "test_func1",
                    "module_name": "mock",
                    "name": "base",
                    "needs": [],
                    "replicas": 1,
                    "resources": [],
                    "show_name": "t1",
                },
                {
                    "cls_name": "",
                    "concurrency": 1,
                    "extra_args": [],
                    "extra_kwargs": {},
                    "func_name": "test_func2",
                    "module_name": "mock",
                    "name": "depend",
                    "needs": ["base"],
                    "replicas": 1,
                    "resources": [],
                    "show_name": "t2",
                },
            ],
        }

        _manifest = load_yaml(bundle_path / DEFAULT_MANIFEST_NAME)
        assert "name" not in _manifest
        assert _manifest["version"] == build_version

        assert m_copy_dir.call_count == 1
        assert m_copy_dir.call_args_list[0][0][0] == "/home/starwhale/myproject"
        assert m_copy_dir.call_args_list[0][0][1].endswith("/src")

        assert bundle_path.exists()
        assert "latest" in sm.tag.list()

        model_uri = URI(f"mnist/version/{build_version}", expected_type=URIType.MODEL)
        sm = StandaloneModel(model_uri)
        _info = sm.info()

        assert _info["basic"]["version"] == build_version
        assert _info["basic"]["name"] == self.name
        assert _info["manifest"]["build"]["os"] in ["Linux", "Darwin"]

        _list, _ = StandaloneModel.list(URI(""))
        assert len(_list) == 1
        assert not _list[self.name][0]["is_removed"]

        model_uri = URI(
            f"{self.name}/version/{build_version}", expected_type=URIType.MODEL
        )
        sd = StandaloneModel(model_uri)
        _ok, _ = sd.remove(False)
        assert _ok

        _list, _ = StandaloneModel.list(URI(""))
        assert _list[self.name][0]["is_removed"]

        _ok, _ = sd.recover(True)
        _list, _ = StandaloneModel.list(URI(""))
        assert not _list[self.name][0]["is_removed"]

        fname = f"{self.name}/version/{build_version}"
        for f in ModelInfoFilter:
            ModelTermView(fname).info(f)
            ModelTermViewJson(fname).info(f)

        ModelTermView(fname).diff(
            URI(fname, expected_type=URIType.MODEL), show_details=False
        )
        ModelTermView(fname).history()
        ModelTermView(fname).remove()
        ModelTermView(fname).recover()
        ModelTermView.list(show_removed=True)
        ModelTermView.list()

        _ok, _ = sd.remove(True)
        assert _ok
        _list, _ = StandaloneModel.list(URI(""))
        assert len(_list[self.name]) == 0

        ModelTermView.build(
            workdir=self.workdir, project="self", model_config=model_config
        )

    def test_get_file_desc(self):
        _file = Path("tmp/file.txt")
        ensure_dir("tmp")
        ensure_file(_file, "123456")
        fd = resource_to_file_node(
            [{"path": "file.txt", "desc": "SRC"}], parent_path=Path("tmp")
        )
        assert "file.txt" in fd
        assert fd.get("file.txt").size == 6

    def test_diff(
        self,
    ) -> None:
        base_version = "base_mdklqrbwpyrb2s3eme63k5xqno4o4wwmd3n"
        compare_version = "compare_mdklqrbwpyrb2s3eme63k5xqno4o4ww"

        base_bundle_path = (
            self.sw.rootdir
            / "self"
            / URIType.MODEL
            / self.name
            / base_version[:VERSION_PREFIX_CNT]
            / f"{base_version}{BundleType.MODEL}"
        )
        ensure_dir(base_bundle_path)
        ensure_file(base_bundle_path / DEFAULT_MANIFEST_NAME, _base_model_yaml)

        compare_bundle_path = (
            self.sw.rootdir
            / "self"
            / URIType.MODEL
            / self.name
            / compare_version[:VERSION_PREFIX_CNT]
            / f"{compare_version}{BundleType.MODEL}"
        )
        ensure_dir(compare_bundle_path)
        ensure_file(compare_bundle_path / DEFAULT_MANIFEST_NAME, _compare_model_yaml)

        base_model_uri = URI(
            f"{self.name}/version/{base_version}", expected_type=URIType.MODEL
        )
        sm = StandaloneModel(base_model_uri)

        diff_info = sm.diff(
            URI(f"{self.name}/version/{compare_version}", expected_type=URIType.MODEL)
        )
        assert len(diff_info) == 3
        # 4 files and 1 added
        assert len(diff_info["all_paths"]) == 5
        assert len(diff_info["base_version"]) == 4
        assert len(diff_info["compare_version"]) == 5
        assert (
            diff_info["compare_version"]["src/model/empty.pt"].flag == FileFlag.UPDATED
        )
        assert diff_info["compare_version"]["src/svc.yaml"].flag == FileFlag.DELETED
        assert diff_info["compare_version"]["src/svc2.yaml"].flag == FileFlag.ADDED
        assert (
            diff_info["compare_version"]["src/runtime.yaml"].flag == FileFlag.UNCHANGED
        )
        assert diff_info["compare_version"]["src/model.yaml"].flag == FileFlag.UNCHANGED

    @patch("starwhale.base.scheduler.Step.get_steps_from_yaml")
    @patch("starwhale.core.model.model.generate_jobs_yaml")
    @patch("starwhale.base.scheduler.Scheduler._schedule_one_task")
    @patch("starwhale.base.scheduler.Scheduler._schedule_one_step")
    @patch("starwhale.base.scheduler.Scheduler._schedule_all")
    def test_run(
        self,
        schedule_all_mock: MagicMock,
        single_step_mock: MagicMock,
        single_task_mock: MagicMock,
        gen_yaml_mock: MagicMock,
        gen_job_mock: MagicMock,
    ):
        gen_job_mock.return_value = [
            Step(
                name="ppl",
                cls_name="",
                resources=[{"type": "cpu", "limit": 1, "request": 1}],
                concurrency=1,
                task_num=1,
                needs=[],
            ),
            Step(
                name="cmp",
                cls_name="",
                resources=[{"type": "cpu", "limit": 1, "request": 1}],
                concurrency=1,
                task_num=1,
                needs=["ppl"],
            ),
        ]
        model_config = ModelConfig(name="test", run={"handlers": ["mock-module"]})
        StandaloneModel.run(
            model_src_dir=Path(self.workdir),
            model_config=model_config,
            project="test",
            version="qwertyuiop",
            dataset_uris=["mnist/version/latest"],
            scheduler_run_args={
                "step_name": "ppl",
                "task_index": 0,
            },
        )
        schedule_all_mock.assert_not_called()
        single_step_mock.assert_not_called()
        single_task_mock.assert_called_once()

        StandaloneModel.run(
            model_src_dir=Path(self.workdir),
            model_config=model_config,
            project="test",
            version="qwertyuiop",
            dataset_uris=["mnist/version/latest"],
            scheduler_run_args={
                "step_name": "ppl",
                "task_index": -1,
            },
        )
        single_step_mock.assert_called_once()

        StandaloneModel.run(
            model_src_dir=Path(self.workdir),
            model_config=model_config,
            project="test",
            version="qwertyuiop",
            dataset_uris=["mnist/version/latest"],
        )
        schedule_all_mock.assert_called_once()

    @Mocker()
    @patch("starwhale.core.model.model.CloudModel.list")
    def test_list_with_project(self, req: Mocker, mock_list: MagicMock):
        base_url = "http://1.1.0.0:8182/api/v1"

        req.request(
            HTTPMethod.POST,
            f"{base_url}/login",
            json={"data": {"name": "foo", "role": {"roleName": "admin"}}},
            headers={"Authorization": "token"},
        )
        InstanceTermView().login(
            "http://1.1.0.0:8182",
            alias="remote",
            username="foo",
            password="bar",
        )
        instances = InstanceTermView().list()
        assert len(instances) == 2  # local and remote

        # default current project is local/self
        models, _ = ModelTermView.list()
        assert len(models) == 0

        mock_models = (
            {
                "proj": [
                    {"version": "foo", "is_removed": False, "created_at": 1},
                    {"version": "bar", "is_removed": False, "created_at": 2},
                ]
            },
            None,
        )
        mock_list.return_value = mock_models
        # list supports using instance/project uri which is not current_instance/current_project
        models, _ = ModelTermView.list("remote/whatever")
        assert len(models) == 2  # project foo and bar

    @patch("starwhale.utils.process.check_call")
    @patch("starwhale.utils.docker.gen_swcli_docker_cmd")
    def test_run_in_container(self, m_gencmd: MagicMock, m_call: MagicMock):
        with tempfile.TemporaryDirectory() as d:
            m_gencmd.return_value = "hi"
            m_call.return_value = 0
            ModelTermView.run_in_container(model_src_dir=Path(d), docker_image="img1")
            m_gencmd.assert_called_once_with("img1", envs={}, mounts=[d])
            m_call.assert_called_once_with("hi", shell=True)

    @patch("starwhale.core.model.model.StandaloneModel")
    @patch("starwhale.core.model.model.StandaloneModel.serve")
    @patch("starwhale.core.runtime.process.Process.from_runtime_uri")
    def test_serve(self, *args: t.Any):
        host = "127.0.0.1"
        port = 80
        yaml_name = "model.yaml"
        runtime = "pytorch/version/latest"
        ModelTermView.serve("", yaml_name, runtime, "mnist/version/latest", host, port)
        ModelTermView.serve(".", yaml_name, runtime, "", host, port)
        ModelTermView.serve(".", yaml_name, "", "", host, port)

        with self.assertRaises(SystemExit):
            ModelTermView.serve("", yaml_name, runtime, "", host, port)

        with self.assertRaises(SystemExit):
            ModelTermView.serve("set", yaml_name, runtime, "set", host, port)


class CloudModelTest(TestCase):
    def setUp(self) -> None:
        sw_config._config = {}

    def test_cli_list(self) -> None:
        mock_obj = MagicMock()
        runner = CliRunner()
        result = runner.invoke(
            list_cli,
            ["--filter", "name=mn", "--filter", "owner=sw", "--filter", "latest"],
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert mock_obj.list.call_count == 1
        call_args = mock_obj.list.call_args[0]
        assert len(call_args[5]) == 3
        assert "name=mn" in call_args[5]
        assert "owner=sw" in call_args[5]
        assert "latest" in call_args[5]

        mock_obj = MagicMock()
        runner = CliRunner()
        result = runner.invoke(
            list_cli,
            [],
            obj=mock_obj,
        )

        assert result.exit_code == 0
        assert mock_obj.list.call_count == 1
        call_args = mock_obj.list.call_args[0]
        assert len(call_args[5]) == 0


@patch("starwhale.core.model.model.generate_jobs_yaml")
@patch("starwhale.core.model.model.StandaloneModel._get_service")
@patch("starwhale.utils.config.load_swcli_config")
def test_build_with_custom_config_file(
    m_sw_config: MagicMock,
    m_get_service: MagicMock,
    m_generate_yaml: MagicMock,
    tmp_path: pathlib.Path,
):
    m_sw_config.return_value = {
        "storage": {
            "root": tmp_path.absolute(),
        },
        "instances": {
            "local": {
                "current_project": "self",
                "type": "standalone",
                "uri": "local",
            }
        },
        "current_instance": "local",
    }

    # generate a fake file as the example
    example = tmp_path / "example.png"
    ensure_file(example, "fake image content")

    svc = MagicMock(spec=Service)
    svc.get_spec.return_value = {}
    svc.example_resources = [example]
    m_get_service.return_value = svc

    name = "foo"
    workdir = tmp_path / "bar"
    cfg = "my_custom_config.yaml"
    ensure_dir(workdir)
    ensure_dir(workdir / "models")
    ensure_file(workdir / "models/mnist_cnn.pt", "baz")
    ensure_dir(workdir / "config")
    ensure_file(workdir / "config/hyperparam.json", "baz")
    ensure_file(workdir / cfg, _model_yaml)

    model_config = ModelConfig.create_by_yaml(workdir / cfg)

    model_uri = URI(name, expected_type=URIType.MODEL)
    sm = StandaloneModel(model_uri)
    sm.build(workdir=workdir, model_config=model_config)

    build_version = sm.uri.object.version

    bundle_path = (
        tmp_path
        / "self"
        / URIType.MODEL
        / name
        / build_version[:VERSION_PREFIX_CNT]
        / f"{build_version}{BundleType.MODEL}"
    )

    assert bundle_path.exists()
    assert (bundle_path / "src").exists()
    assert (bundle_path / "src" / DefaultYAMLName.MODEL).exists()
    assert (bundle_path / "src" / ".starwhale" / "examples" / "example.png").exists()


@patch("starwhale.core.model.model.generate_jobs_yaml")
@patch("starwhale.utils.config.load_swcli_config")
def test_render_eval_layout(m_sw_config: MagicMock, m_g: MagicMock, tmp_path: Path):
    m_sw_config.return_value = {
        "storage": {
            "root": tmp_path.absolute(),
        },
        "instances": {
            "local": {
                "current_project": "self",
                "type": "standalone",
                "uri": "local",
            }
        },
        "current_instance": "local",
    }
    name = "bar"
    model_uri = URI(name, expected_type=URIType.MODEL)
    sm = StandaloneModel(model_uri)

    workdir = tmp_path / name
    model_config = ModelConfig(name=name, run={"handler": "baz:main"})
    ensure_file(workdir / "baz.py", "def main(): pass", parents=True)

    # test with no layout
    # no error should be raised
    sm.build(workdir=workdir, model_config=model_config)

    build_version = sm.uri.object.version
    bundle_path = (
        tmp_path
        / "self"
        / URIType.MODEL
        / name
        / build_version[:VERSION_PREFIX_CNT]
        / f"{build_version}{BundleType.MODEL}"
    )
    empty_dir(bundle_path)

    # test with json layout
    d = workdir / SW_AUTO_DIRNAME / EVALUATION_PANEL_LAYOUT_JSON_FILE_NAME
    ensure_file(d, '{"foo": "bar"}', parents=True)
    sm.build(workdir=workdir, model_config=model_config)

    layout = (
        bundle_path / "src" / SW_AUTO_DIRNAME / EVALUATION_PANEL_LAYOUT_JSON_FILE_NAME
    )
    assert layout.exists()
    assert layout.read_text() == '{"foo": "bar"}'
    empty_dir(bundle_path)

    # test with yaml layout
    d = workdir / SW_AUTO_DIRNAME / EVALUATION_PANEL_LAYOUT_YAML_FILE_NAME
    ensure_file(d, "bar: baz", parents=True)
    sm.build(workdir=workdir, model_config=model_config)
    assert layout.exists()
    assert layout.read_text() == '{"bar": "baz"}'
