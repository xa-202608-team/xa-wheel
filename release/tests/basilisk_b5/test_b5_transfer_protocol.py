"""tests/basilisk_b5/test_b5_transfer_protocol.py —— §3/§4/§16 迁移协议纪律。

- 只有三个学习组 + 两个只评估组, 不得加第四种迁移方法 (§4)
- MMD 的 lambda 保持冻结值, 禁止 grid search / 逐 seed lambda / 在 test 上调 (§16)
- 协议必须在任何 B5 数字之前冻结, 事后不得改动 (§2/§7)
"""
from __future__ import annotations

import hashlib

from conftest import ROOT, code_nospace, code_only

TRANSFER_SCRIPTS = (
    "scripts/basilisk_b5/run_formal_transfer.py",
    "scripts/basilisk_b5/data_b5.py",
    "scripts/basilisk_b5/analyze_paired_gain.py",
    "scripts/basilisk_b5/analyze_lifetime_bins.py",
    "scripts/basilisk_b5/analyze_warning_metrics.py",
    "scripts/basilisk_b5/summarize_b5.py",
)


def test_b5_protocol_frozen_before_any_number(b5_protocol_hash):
    assert bool(b5_protocol_hash["frozen_before_any_b5_number"]) is True
    assert len(str(b5_protocol_hash["protocol_sha256"])) == 64
    assert len(str(b5_protocol_hash["config_sha256"])) == 64


def test_b5_protocol_hash_still_matches(b5_protocol_hash, b5_config):
    """protocol.md 在冻结之后不得改动 —— 改了就是事后改门槛。"""
    p = ROOT / b5_config["protocol"]["path"]
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    assert got == str(b5_protocol_hash["protocol_sha256"]), \
        "protocol.md 在冻结后被改动 —— B5_INVALID"


def test_b5_metrics_reference_frozen_protocol(b5_metrics, b5_protocol_hash):
    assert str(b5_metrics["protocol_sha256"]) == \
        str(b5_protocol_hash["protocol_sha256"])
    assert str(b5_metrics["config_sha256"]) == \
        str(b5_protocol_hash["config_sha256"])


def test_b5_mmd_lambda_frozen(b5_metrics, b5_config, b5_contract):
    """test_b5_mmd_lambda_frozen (§21/§16) —— lambda 与 B2 冻结值一致。"""
    num = {}
    for k, v in b5_contract.items():
        if k.endswith("_results") and isinstance(v, dict):
            num.update(v)
    lam = float(b5_metrics["mmd_lambda"])
    assert lam == float(num["mmd_lambda"]) == 1.0
    assert lam == float(b5_config["transfer"]["mmd_lambda"])
    assert bool(b5_metrics["mmd_lambda_frozen"]) is True
    # 逐 seed 记录里也必须是同一个 lambda —— 不允许 seed-specific lambda
    for r in b5_metrics["per_seed"]:
        st = r["source_mmd_finetune_meta"]["stages"]
        lams = [float(s["mmd_lambda"]) for s in st if "mmd_lambda" in s]
        assert lams, "MMD 组未记录 lambda"
        assert set(lams) == {lam}, f"seed {r['seed']} 出现了不同的 lambda: {lams}"


def test_b5_no_mmd_grid_search():
    """§16: 绝对禁止 lambda 网格搜索 / 重新挑源域层。"""
    for rel in TRANSFER_SCRIPTS:
        src = code_nospace(rel)
        for bad in ("lambda_grid", "mmd_lambda_candidates", "lambda_sweep", "forlamin", "itertools.product"):
            assert bad not in src, f"{rel} 出现疑似 lambda 搜索: {bad}"


def test_b5_only_three_trained_groups(b5_metrics, b5_runner):
    assert tuple(b5_runner.TRAINED_GROUPS) == (
        "target_only", "source_finetune", "source_mmd_finetune")
    assert tuple(b5_runner.EVAL_ONLY_GROUPS) == (
        "const_mean_info", "damage_extrapolation")
    assert set(b5_metrics["trained_groups"]) == set(b5_runner.TRAINED_GROUPS)
    assert set(b5_metrics["eval_only_groups"]) == set(b5_runner.EVAL_ONLY_GROUPS)


def test_b5_source_checkpoint_unchanged(b5_metrics, b5_contract):
    """迁移起点必须是那一份源域权重 —— 不得替换、不得重训。"""
    num = {}
    for k, v in b5_contract.items():
        if k.endswith("_results") and isinstance(v, dict):
            num.update(v)
    assert str(b5_metrics["source_checkpoint_sha256"]) == \
        num["source_checkpoint_sha256"]
    for r in b5_metrics["per_seed"]:
        for g in ("source_finetune", "source_mmd_finetune"):
            meta = r[f"{g}_meta"]
            assert bool(meta["source_checkpoint_loaded"]) is True
            assert str(meta["source_checkpoint_sha256"]) == \
                num["source_checkpoint_sha256"]
        assert bool(r["target_only_meta"]["source_checkpoint_loaded"]) is False


def test_b5_target_only_does_not_load_source(b5_metrics):
    """target_only 必须是随机/目标域初始化 —— 否则三组的唯一差异被抹掉。"""
    for r in b5_metrics["per_seed"]:
        assert bool(r["target_only_meta"]["init_from_source_checkpoint"]) is False
        assert bool(r["source_finetune_meta"]["init_from_source_checkpoint"]) \
            is True


def test_b5_only_mmd_group_uses_mmd(b5_metrics):
    for r in b5_metrics["per_seed"]:
        assert bool(r["source_mmd_finetune_meta"]["use_mmd"]) is True
        assert bool(r["source_finetune_meta"]["use_mmd"]) is False
        assert bool(r["target_only_meta"]["use_mmd"]) is False


def test_b5_transfer_happens_at_hi_dynamics_layer():
    """迁移必须在 HI / 退化动力学层, 不在原始波形层。"""
    src = code_only("scripts/basilisk_b5/run_formal_transfer.py")
    assert "load_pretrained" in src, "未走 encoder 权重迁移"
    for bad in ("raw_waveform", "vibration_raw", "resample_waveform"):
        assert bad not in src, f"出现原始波形层迁移痕迹: {bad}"


def test_b5_source_and_target_generators_isolated(b5_metrics, b5_data_mod):
    """源域 loader 与目标域 loader 的随机源必须隔离 (§6.1)。"""
    assert b5_data_mod.TARGET_GEN_OFFSET == 7
    for r in b5_metrics["per_seed"]:
        s = int(r["seed"])
        for g in ("target_only", "source_finetune", "source_mmd_finetune"):
            m = r[f"{g}_meta"]
            assert int(m["batch_order_generator_seed"]) == s * 1000 + 7
            assert int(m["source_generator_seed"]) == s * 1000 + 11
            assert m["batch_order_generator_seed"] != m["source_generator_seed"]


def test_b5_no_llm_in_numeric_path():
    """LLM 不参与数值寿命预测。"""
    for rel in TRANSFER_SCRIPTS:
        src = code_nospace(rel).lower()
        for bad in ("openai", "llm_predict", "chatcompletion"):
            assert bad not in src, f"{rel} 出现 LLM 调用: {bad}"


def test_b5_no_global_numpy_seed():
    """禁止全局 np.random.seed() —— 只允许显式 Generator。"""
    for rel in TRANSFER_SCRIPTS:
        src = code_nospace(rel)
        assert "np.random.seed" not in src, f"{rel} 使用了全局 numpy 种子"
