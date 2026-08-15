# -*- coding: utf-8 -*-
"""verify_manifest.py 阶段二"磁盘->manifest"多余文件检测的合成探针测试。

用临时目录构造最小 HANDOFF_MANIFEST.json + 线下工件：
- 正向：装配目录与登记完全一致 -> verify_artifact_root 通过；
- 反向：目录中混入一个未登记的多余文件 -> FAIL 且 EXTRA 指认该文件；
- 回归：既有校验不降低 —— 登记文件被删除（MISSING）/字节被改（HASH）仍 FAIL。

只测阶段二函数本身，不依赖仓库 CODE_MANIFEST 状态，可独立复跑。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERIFY_SCRIPT = _REPO_ROOT / "release" / "scripts" / "handoff" / "verify_manifest.py"


def _load_verify_manifest():
    spec = importlib.util.spec_from_file_location("xa_wheel_verify_manifest", _VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_entry(rel: str, payload: bytes) -> dict:
    return {
        "path": rel,
        "role": "checkpoint",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.fixture()
def artifact_root(tmp_path: Path):
    """最小合法装配目录：HANDOFF_MANIFEST.json 登记 2 个工件且字节一致。"""
    (tmp_path / "checkpoints").mkdir()
    payload_a = b"pretend-source-pretrain-weights" * 8
    payload_b = b"pretend-b6-summary-json" * 16
    (tmp_path / "checkpoints" / "source_tcn_pretrain.pt").write_bytes(payload_a)
    (tmp_path / "checkpoints" / "summary.json").write_bytes(payload_b)
    doc = {
        "schema_version": "1.1.0",
        "component": "wheel",
        "files": [
            _make_entry("checkpoints/source_tcn_pretrain.pt", payload_a),
            _make_entry("checkpoints/summary.json", payload_b),
        ],
    }
    (tmp_path / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp_path


def test_phase2_passes_when_disk_matches_manifest(artifact_root: Path) -> None:
    """正向：无多余文件时阶段二整体通过（含新的磁盘->manifest 方向）。"""
    mod = _load_verify_manifest()
    assert mod.verify_artifact_root(artifact_root) is True


def test_phase2_fails_on_undeclared_extra_file(artifact_root: Path) -> None:
    """反向：装配目录混入未登记文件 -> FAIL，且 EXTRA 指认该文件。"""
    sneaky = artifact_root / "checkpoints" / "unaudited_extra.json"
    sneaky.write_bytes(b"{\"not\": \"declared\"}")
    mod = _load_verify_manifest()
    assert mod.verify_artifact_root(artifact_root) is False


def test_phase2_extra_detection_catches_nested_and_root_files(artifact_root: Path) -> None:
    """反向：根下与深层的未登记文件都算 EXTRA（逐条列举前 15 条）。"""
    (artifact_root / "root_extra.txt").write_bytes(b"x")
    deep_dir = artifact_root / "data" / "sim" / "deep"
    deep_dir.mkdir(parents=True)
    (deep_dir / "nested_extra.h5").write_bytes(b"y")
    mod = _load_verify_manifest()
    assert mod.verify_artifact_root(artifact_root) is False


def test_phase2_existing_checks_not_reduced(artifact_root: Path) -> None:
    """回归：既有校验保持 —— MISSING（登记文件消失）与 HASH（字节被改）仍 FAIL。"""
    mod = _load_verify_manifest()

    # MISSING: 删掉一个登记文件
    victim = artifact_root / "checkpoints" / "summary.json"
    backup = victim.read_bytes()
    victim.unlink()
    assert mod.verify_artifact_root(artifact_root) is False

    # HASH: 恢复文件但篡改字节（size 同步改, 隔离 HASH 一向）
    tampered = backup + b"!"
    victim.write_bytes(tampered)
    doc = json.loads((artifact_root / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8"))
    for ent in doc["files"]:
        if ent["path"] == "checkpoints/summary.json":
            ent["size"] = len(tampered)
    (artifact_root / "HANDOFF_MANIFEST.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    assert mod.verify_artifact_root(artifact_root) is False
