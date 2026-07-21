"""Run YOLOv5 training only after the selected dataset passes the audit."""

import os
import json
import hashlib
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from dataset_audit import audit_dataset
from pretrained_weights import validate_training_weight
from project_paths import PRETRAINED_WEIGHTS_DIR, PROJECT_ROOT, TRAINING_OUTPUTS_DIR


def option_value(arguments, option):
    value = None
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            candidate = arguments[index + 1]
            if not candidate.startswith("--"):
                value = candidate
        prefix = option + "="
        if argument.startswith(prefix):
            value = argument[len(prefix):]
    return value


def has_option(arguments, option):
    return any(argument == option or argument.startswith(option + "=") for argument in arguments)


def has_abbreviated_option(arguments, option):
    for argument in arguments:
        name = argument.split("=", 1)[0]
        if name.startswith("--") and name != option and option.startswith(name):
            return True
    return False


def remove_option(arguments, option):
    """Remove every occurrence so a later duplicate cannot bypass validation."""
    cleaned = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            index += 2
            continue
        if argument.startswith(option + "="):
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned


def resolve_from_project(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _portable_display_path(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resume_candidate(value):
    checkpoint = resolve_from_project(value)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "weights" / "last.pt"
    if not checkpoint.is_file() or checkpoint.name.lower() != "last.pt":
        print("拒绝续训：请选择候选实验中的 weights/last.pt。")
        return 2
    if not is_within(checkpoint, TRAINING_OUTPUTS_DIR.resolve()):
        print("拒绝续训：checkpoint 必须位于 TRAINING_OUTPUTS。")
        return 2

    run_directory = checkpoint.parent.parent
    opt_path = run_directory / "opt.yaml"
    manifest_path = run_directory / "candidate_manifest.json"
    if not opt_path.is_file():
        print("拒绝续训：候选实验缺少 opt.yaml。")
        return 2
    with opt_path.open("r", encoding="utf-8-sig") as handle:
        options = yaml.safe_load(handle) or {}

    pending_manifest_path = TRAINING_OUTPUTS_DIR / "_MANIFESTS" / (run_directory.name + ".json")
    manifest = {}
    manifest_source = manifest_path if manifest_path.is_file() else pending_manifest_path
    if manifest_source.is_file():
        with manifest_source.open("r", encoding="utf-8-sig") as handle:
            manifest = json.load(handle) or {}
    dataset_value = manifest.get("dataset_yaml_relative") or options.get("data")
    if not dataset_value:
        print("拒绝续训：无法确定原训练数据集 YAML。")
        return 2
    dataset_path = resolve_from_project(dataset_value)
    if not dataset_path.is_file():
        print("拒绝续训：原数据集配置在当前工作库中不存在：%s" % dataset_path)
        return 2
    expected_config_hash = str(manifest.get("dataset_yaml_sha256") or "").lower()
    actual_config_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if expected_config_hash and actual_config_hash != expected_config_hash:
        print("拒绝续训：数据集 YAML 自初训后已发生变化。")
        print("如需使用新数据定义，请开始一个新的候选实验，避免混合实验版本。")
        return 2
    if audit_dataset(str(dataset_path)) != 0:
        print("拒绝续训：当前数据集审计未通过。")
        return 2

    # YOLOv5 reads opt.yaml during resume.  Refresh only the portable data
    # path and preserve the original file once for auditability.
    backup_path = run_directory / "opt.original.yaml"
    if not backup_path.exists():
        shutil.copy2(str(opt_path), str(backup_path))
    options["data"] = str(dataset_path)
    options["project"] = str(TRAINING_OUTPUTS_DIR.resolve())
    options["save_dir"] = str(run_directory)
    options["name"] = run_directory.name
    temporary_opt = opt_path.with_suffix(".yaml.tmp")
    with temporary_opt.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(options, handle, allow_unicode=True, sort_keys=False)
    temporary_opt.replace(opt_path)

    manifest.update({
        "schema_version": 1,
        "status": "resuming",
        "dataset_yaml": str(dataset_path),
        "dataset_yaml_relative": _portable_display_path(dataset_path),
        "dataset_yaml_sha256": actual_config_hash,
        "run_directory": str(run_directory),
        "run_directory_relative": _portable_display_path(run_directory),
    })
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--resume", str(checkpoint)]
    print("数据集审计通过，正在安全续训：%s" % run_directory)
    exit_code = subprocess.call(command, cwd=str(PROJECT_ROOT))
    if manifest_path.is_file():
        history = manifest.setdefault("resume_history", [])
        history.append({
            "at": datetime.now().astimezone().isoformat(),
            "checkpoint": _portable_display_path(checkpoint),
            "exit_code": exit_code,
        })
        manifest["status"] = "completed" if exit_code == 0 else "failed_or_stopped"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        if pending_manifest_path.is_file():
            try:
                pending_manifest_path.unlink()
            except OSError:
                pass
    return exit_code


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    safe_resume = option_value(arguments, "--safe-resume")
    if safe_resume:
        allowed = remove_option(arguments, "--safe-resume")
        if allowed:
            print("拒绝续训：--safe-resume 不能与其他训练参数混用。")
            return 2
        return resume_candidate(safe_resume)
    if has_option(arguments, "--resume") or has_abbreviated_option(arguments, "--resume"):
        print("拒绝自动 resume：旧训练记录可能引用已变更的数据配置。")
        print("需要续训时，请先核对该实验的 opt.yaml 和数据版本。")
        return 2

    if has_option(arguments, "--exist-ok") or has_abbreviated_option(arguments, "--exist-ok"):
        print("拒绝 --exist-ok：候选实验不得覆盖或混入旧目录。")
        return 2

    for guarded_option in ("--project", "--name", "--data", "--weights"):
        if has_abbreviated_option(arguments, guarded_option):
            print("拒绝缩写参数 %s：安全路径参数必须写完整。" % guarded_option)
            return 2

    data_config = option_value(arguments, "--data")
    if not data_config:
        print("拒绝启动训练：必须明确提供 --data 数据集 YAML。")
        print("示例: TRAIN_SAFE.cmd --data data/new_target.yaml --weights PRETRAINED_WEIGHTS/yolov5s.pt --epochs 100 --batch-size -1")
        return 2
    data_path = resolve_from_project(data_config)
    if not data_path.is_file():
        print("拒绝启动训练：数据集 YAML 不存在：%s" % data_path)
        return 2

    weights_value = option_value(arguments, "--weights")
    if weights_value is None:
        weights_path = PRETRAINED_WEIGHTS_DIR / "yolov5s.pt"
    elif weights_value:
        weights_path = resolve_from_project(weights_value)
    else:
        weights_path = None
    if weights_path is not None:
        if not weights_path.is_file():
            print("拒绝启动训练：初始权重必须是工作库中已存在的本地文件。")
            print("离线工作库不会自动从网络下载缺失权重：%s" % weights_path)
            return 2
        weight_ok, weight_message = validate_training_weight(weights_path)
        if not weight_ok:
            print("拒绝启动训练：%s" % weight_message)
            return 2
        print("[OK] %s" % weight_message)

    requested_project = option_value(arguments, "--project") or str(TRAINING_OUTPUTS_DIR)
    output_project = resolve_from_project(requested_project)
    if not is_within(output_project, TRAINING_OUTPUTS_DIR.resolve()):
        print("拒绝启动训练：训练产物必须保存在 TRAINING_OUTPUTS 候选区。")
        print("RESULTS 只能存放经人工检查并明确批准的 SENTINEL 部署模型。")
        print("收到的输出路径: %s" % output_project)
        return 2

    if audit_dataset(str(data_path)) != 0:
        print("\n拒绝启动训练：数据集审计未通过。")
        return 2

    # Reinsert every validated safety-critical path.  This also removes
    # duplicates so argparse can never train with a different later value.
    arguments = remove_option(arguments, "--data")
    arguments.extend(["--data", str(data_path)])
    arguments = remove_option(arguments, "--weights")
    arguments.extend(["--weights", str(weights_path) if weights_path is not None else ""])

    arguments = remove_option(arguments, "--project")
    arguments.extend(["--project", str(output_project)])

    requested_name = option_value(arguments, "--name") or "candidate"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", requested_name).strip("_") or "candidate"
    unique_name = "%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), safe_name)
    run_directory = output_project / unique_name
    if run_directory.exists():
        print("拒绝启动训练：候选目录已存在：%s" % run_directory)
        return 2
    arguments = remove_option(arguments, "--name")
    arguments.extend(["--name", unique_name])

    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ["SENTINEL_OFFLINE"] = "1"
    command = [sys.executable, str(PROJECT_ROOT / "train.py")] + arguments
    print("\n数据集审计已通过，正在启动 YOLOv5 训练。")
    print("未审核候选模型目录: %s" % run_directory)
    print("训练完成后不会自动替换 RESULTS 中的 SENTINEL 模型。")
    config_path = data_path
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "dataset_yaml": str(config_path),
        "dataset_yaml_relative": _portable_display_path(config_path),
        "dataset_yaml_sha256": config_hash,
        "run_directory": str(run_directory),
        "run_directory_relative": _portable_display_path(run_directory),
        "command": command,
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "computer": platform.node(),
        },
    }
    manifest_directory = TRAINING_OUTPUTS_DIR / "_MANIFESTS"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    pending_manifest = manifest_directory / (unique_name + ".json")
    with pending_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT))
    local_manifest_written = False
    while process.poll() is None:
        if run_directory.is_dir() and not local_manifest_written:
            with (run_directory / "candidate_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            local_manifest_written = True
        time.sleep(0.25)
    exit_code = process.returncode
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["exit_code"] = exit_code
    manifest["status"] = "completed" if exit_code == 0 else "failed_or_stopped"
    final_manifest = run_directory / "candidate_manifest.json"
    if run_directory.is_dir():
        with final_manifest.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        try:
            pending_manifest.unlink()
        except OSError:
            pass
    else:
        with pending_manifest.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
