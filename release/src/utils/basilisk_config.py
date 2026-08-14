"""utils/basilisk_config.py

Basilisk-v1 配置加载 —— configs/wheel_basilisk.yaml 通过 base_config 继承
configs/wheel.yaml, 深度合并后返回。

为什么必须继承而不是复制:
  退化物理参数 (physics / disturbance / failure / hi) 只能有**一份**定义。若把它们
  复制进 wheel_basilisk.yaml, 两份数值会独立漂移, 之后 "analytic vs basilisk 对种"
  (§14) 就分不清差异来自工况还是来自物理参数, 整个对比失去意义。

  同时这也保证 §2 的"不改旧代码": wheel.yaml 一个字节都不动 (契约已冻结它的 hash)。
"""
from __future__ import annotations

import copy
from pathlib import Path

from .config import PROJECT_ROOT, load_config


def deep_merge(base: dict, over: dict) -> dict:
    """递归合并; over 的标量/列表整体替换 base 的同名键。

    列表**不做逐元素合并**: mode 列表、hi_bins 这类配置一旦半合并会得到语义上
    不存在的中间值, 整体替换才是可预期的行为。
    """
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_basilisk_config(path: str | Path = "configs/wheel_basilisk.yaml") -> dict:
    """加载 Basilisk-v1 配置 (含 base_config 继承)。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    over = load_config(p)
    base_rel = over.pop("base_config", None)
    if base_rel is None:
        return over
    base = load_config(PROJECT_ROOT / base_rel)
    cfg = deep_merge(base, over)
    cfg["_base_config"] = str(base_rel)
    cfg["_config_path"] = str(p.relative_to(PROJECT_ROOT).as_posix())
    return cfg
