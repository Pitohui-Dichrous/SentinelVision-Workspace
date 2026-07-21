"""Beginner-friendly, read-only environment check for SentinelVision."""

import importlib
import contextlib
import io
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from pretrained_weights import PRESETS
from project_paths import PROJECT_ROOT, RUNTIME_DIR, RUNTIME_PYTHON


os.environ["YOLOv5_AUTOINSTALL"] = "false"
os.environ["SENTINEL_OFFLINE"] = "1"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


REQUIRED_MODULES = {
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "cv2": "OpenCV",
    "PySide6": "PySide6",
    "numpy": "NumPy",
    "yaml": "PyYAML",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
    "PIL": "Pillow",
    "scipy": "SciPy",
    "seaborn": "Seaborn",
    "tensorboard": "TensorBoard",
    "tqdm": "tqdm",
    "thop": "THOP",
    "requests": "Requests",
    "psutil": "psutil",
    "IPython": "IPython",
    "git": "GitPython",
}


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except (OSError, ValueError):
        return False


def _ok(message):
    print("[OK] %s" % message)


def _error(message):
    print("[错误] %s" % message)


def _warning(message):
    print("[提示] %s" % message)


def _nvidia_smi_summary():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _check_dynamic_models():
    """Scan RESULTS and isolate bad models without blocking healthy ones."""
    try:
        from model_catalog import ModelCatalog

        snapshot = ModelCatalog().scan(verify_hashes=True, inspect_new_models=True)
    except Exception as exc:
        _error("动态模型目录无法扫描: %s" % exc)
        return ["动态模型扫描"]

    for warning in snapshot.warnings:
        _warning(warning)
    for model in snapshot.models:
        size_mb = model.weight_path.stat().st_size / (1024 ** 2) if model.weight_path.is_file() else 0.0
        if model.selectable:
            source = "已登记" if model.source == "registry" else "新发现"
            _ok("模型 %-16s %-6s %s (%.1f MB)" % (
                model.model_id, source, model.relative_weight_path, size_mb
            ))
            if model.source == "discovered":
                _warning("%s 尚未登记；未知类别默认不触发告警。" % model.display_name)
        else:
            _warning("已隔离模型 %s: %s" % (model.display_name, model.issue or "不可用"))

    if not snapshot.selectable_models:
        _error("RESULTS 中没有可用模型。")
        return ["可用部署模型"]
    _ok("动态模型扫描完成：%d 个可用，单个损坏模型不会影响其余模型" % len(snapshot.selectable_models))
    return []


def run_check():
    failures = []
    print("SentinelVision 环境自检")
    print("=" * 60)
    print("项目目录: %s" % PROJECT_ROOT)
    print("Python: %s" % sys.executable)
    print("Python 版本: %s" % platform.python_version())

    try:
        actual_python = Path(sys.executable).resolve()
        expected_python = RUNTIME_PYTHON.resolve()
        if actual_python != expected_python:
            failures.append("便携 Python")
            _error("当前解释器不是工作库内置 Python: %s" % actual_python)
        elif not _is_within(sys.prefix, RUNTIME_DIR) or not _is_within(sys.base_prefix, RUNTIME_DIR):
            failures.append("便携 Python 前缀")
            _error("Python 前缀不在工作库 RUNTIME 内。")
        elif sys.prefix != sys.base_prefix:
            failures.append("Python 虚拟环境")
            _error("检测到依赖宿主机的虚拟环境；成品必须使用独立便携 Python。")
        else:
            _ok("独立便携 Python 已从工作库 RUNTIME 启动")
    except OSError as exc:
        failures.append("便携 Python 路径")
        _error("无法核对便携 Python 路径: %s" % exc)

    if struct.calcsize("P") * 8 == 64:
        _ok("Python 架构: 64 位")
    else:
        failures.append("Python 架构")
        _error("项目要求 64 位 Windows Python。")

    try:
        path_file = RUNTIME_DIR / "python" / "python311._pth"
        path_text = path_file.read_text(encoding="ascii")
        if ":\\" in path_text or ":/" in path_text or "..\\.." not in path_text:
            raise ValueError("python311._pth 含固定盘符或缺少项目相对路径")
        if (RUNTIME_DIR / "python" / "pyvenv.cfg").exists():
            raise ValueError("检测到不可移植的 pyvenv.cfg")
        external_paths = [
            item for item in sys.path
            if item and Path(item).is_absolute() and not _is_within(item, PROJECT_ROOT)
        ]
        if external_paths:
            raise ValueError("sys.path 含工作库外路径: %s" % external_paths[0])
        _ok("Python 搜索路径为相对、隔离且不依赖宿主环境")
    except (OSError, ValueError) as exc:
        failures.append("Python 隔离路径")
        _error("便携 Python 路径配置异常: %s" % exc)

    try:
        manifest_path = RUNTIME_DIR / "bootstrap" / "runtime_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        entries = manifest.get("files", [])
        wheel_entries = [entry for entry in entries if str(entry.get("path", "")).startswith("wheelhouse/")]
        if manifest.get("schema_version") != 1 or manifest.get("python_version") != "3.11.9":
            raise ValueError("恢复清单版本异常")
        if len(entries) < 80 or len(wheel_entries) < 75:
            raise ValueError("恢复清单条目不足")
        for entry in wheel_entries:
            if not (RUNTIME_DIR / entry["path"]).is_file():
                raise FileNotFoundError(entry["path"])
        _ok("离线恢复清单与 wheelhouse 文件齐备")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append("离线恢复包")
        _error("离线恢复包不完整: %s" % exc)

    try:
        runtime_directory = PROJECT_ROOT / ".runtime"
        runtime_directory.mkdir(parents=True, exist_ok=True)
        probe = runtime_directory / (".write_check_%s.tmp" % os.getpid())
        with probe.open("x", encoding="ascii") as handle:
            handle.write("ok")
        probe.unlink()
        _ok("工作库可写")
    except OSError as exc:
        failures.append("工作库写权限")
        _error("工作库不可写；无法训练或保存设置: %s" % exc)

    try:
        free_gb = shutil.disk_usage(str(PROJECT_ROOT)).free / (1024 ** 3)
        if free_gb < 5:
            failures.append("剩余空间")
            _error("工作库所在磁盘仅剩 %.1f GB，无法安全训练" % free_gb)
        elif free_gb < 20:
            _warning("工作库所在磁盘仅剩 %.1f GB，较长训练前请清理空间" % free_gb)
        else:
            _ok("工作库剩余空间: %.1f GB" % free_gb)
    except OSError as exc:
        _warning("无法读取工作库剩余空间: %s" % exc)

    if sys.version_info[:3] == (3, 11, 9):
        _ok("Python 版本: 3.11.9（工作库固定版本）")
    else:
        failures.append("Python 版本")
        _error("工作库要求内置 Python 3.11.9。")

    imported = {}
    for module_name, display_name in REQUIRED_MODULES.items():
        try:
            imported[module_name] = importlib.import_module(module_name)
            version = getattr(imported[module_name], "__version__", "已安装")
            _ok("%s: %s" % (display_name, version))
        except Exception as exc:  # DLL errors are as important as missing modules.
            failures.append(display_name)
            _error("%s 不可用: %s" % (display_name, exc))

    try:
        if importlib.util.find_spec("pkg_resources") is None:
            raise ImportError("pkg_resources not found")
        _ok("Setuptools/pkg_resources 可用")
    except Exception as exc:
        failures.append("Setuptools")
        _error("Setuptools/pkg_resources 不可用: %s" % exc)

    for module_name, module in imported.items():
        module_file = getattr(module, "__file__", None)
        if module_file and not _is_within(module_file, RUNTIME_DIR):
            failures.append("%s 来源" % REQUIRED_MODULES[module_name])
            _error("%s 错误地来自工作库外部: %s" % (REQUIRED_MODULES[module_name], module_file))

    try:
        importlib.import_module("models.common")
        _ok("YOLOv5 本地模型加载链可用")
    except Exception as exc:
        failures.append("YOLOv5 模型加载链")
        _error("YOLOv5 本地模型加载链不可用: %s" % exc)

    torch = imported.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                _ok("CUDA 可用: %s (%.1f GB), PyTorch CUDA %s" % (device_name, memory_gb, torch.version.cuda))
            else:
                failures.append("CUDA")
                _error("PyTorch 无法使用 CUDA，训练会退化为 CPU")
        except Exception as exc:
            failures.append("CUDA")
            _error("CUDA 检查失败: %s" % exc)
    else:
        gpu_summary = _nvidia_smi_summary()
        if gpu_summary:
            _warning("显卡驱动可见，但 PyTorch 尚未正确安装: %s" % gpu_summary)

    if imported.get("yaml") is not None and imported.get("torch") is not None:
        failures.extend(_check_dynamic_models())

    missing_weights = []
    damaged_weights = []
    for preset in PRESETS:
        if not preset.path.is_file():
            missing_weights.append(preset.filename)
        elif preset.path.stat().st_size != preset.size_bytes:
            damaged_weights.append(preset.filename)
    if missing_weights or damaged_weights:
        failures.append("官方预训练权重")
        if missing_weights:
            _error("缺少预训练权重: %s" % ", ".join(missing_weights))
        if damaged_weights:
            _error("预训练权重大小异常: %s" % ", ".join(damaged_weights))
    else:
        _ok("YOLOv5 v7.0 官方目标检测权重齐备: 10 个（P5 + P6）")

    required_project_files = [
        "train.py", "detect.py", "project_paths.py", "launcher_GPT.py",
        "SentinelVision4.py", "model_catalog.py", "model_runtime.py",
        "workspace_manager.py", "deployment_manager.py", "safe_train.py",
        "pretrained_weights.py", "ui_theme.py",
        "dataset_audit.py", "portable/setup_environment.ps1",
        "portable/runtime_env.cmd",
        "portable/verify_workspace.py", "VERIFY_WORKSPACE.cmd",
        "requirements.txt", "requirements-torch-cu118.txt",
        "ui_font.py", "START_HERE.cmd", "START_HERE_CN.md",
        "assets/NotoSansSC-VF.ttf", "assets/NotoSansSC-LICENSE.txt",
        "RESULTS/model_registry.yaml",
        "PRETRAINED_WEIGHTS/yolov5s.pt",
        "DATASETS/combined_legacy_v1/dataset.yaml",
        "DATASETS/fire_only_legacy_v1/dataset.yaml",
        "RUNTIME/python/python.exe",
        "RUNTIME/python/python311._pth",
        "RUNTIME/bootstrap/python-3.11.9-embed-amd64.zip",
        "RUNTIME/bootstrap/get-pip.py",
        "RUNTIME/bootstrap/requirements-lock.txt",
        "RUNTIME/bootstrap/runtime_manifest.json",
    ]
    for relative_path in required_project_files:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            failures.append(relative_path)
            _error("缺少项目文件: %s" % relative_path)

    # These two generated views are shipped as known-good examples and as
    # the default training choices.  Auditing their contents also detects an
    # interrupted copy to a removable drive, not merely a copied YAML file.
    if not any(item in failures for item in ("dataset_audit.py", "DATASETS/combined_legacy_v1/dataset.yaml", "DATASETS/fire_only_legacy_v1/dataset.yaml")):
        try:
            from dataset_audit import audit_dataset
            for relative_path in (
                "DATASETS/combined_legacy_v1/dataset.yaml",
                "DATASETS/fire_only_legacy_v1/dataset.yaml",
            ):
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    audit_result = audit_dataset(
                        str(PROJECT_ROOT / relative_path),
                        verify_images=False,
                        verify_split_content=False,
                    )
                if audit_result != 0:
                    print(captured.getvalue())
                    failures.append(relative_path + " 内容")
                else:
                    _ok("内置数据集完整: %s" % relative_path)
        except Exception as exc:
            failures.append("内置数据集完整性")
            _error("内置数据集完整性检查失败: %s" % exc)

    print("=" * 60)
    if failures:
        _error("自检未通过，共 %d 项问题: %s" % (len(failures), ", ".join(failures)))
        print("如果是依赖问题，请双击 SETUP_ENVIRONMENT.cmd。")
        return 1

    _ok("自检通过，可以启动前端或进行训练。")
    return 0


if __name__ == "__main__":
    sys.exit(run_check())
