"""迁移不变量测试 (plan P5 验收 + S3 协议 §18):
  - S2 编码器冻结 / S3 解冻
  - 轨迹级划分不相交且覆盖
  - MMD 按 HI 健康阶段分箱 (非按时间)
  - S3 删失损失 (观测 Huber / one-sided hinge / 梯度) 与三组公平性、早停不触碰 test
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load_config                                       # noqa: E402
from src.transfer.adapter import TransferModel                          # noqa: E402
from src.transfer.mmd import mmd_by_hi_bins                             # noqa: E402
from src.transfer.train_transfer import split_trajectories              # noqa: E402
from src.experiments import run_groups as rg                            # noqa: E402
import scripts.run_s3_gate as s3g                                       # noqa: E402


def test_encoder_freeze_in_S2_unfreeze_in_S3():
    """S2 编码器参数 requires_grad=False (冻结); S3 解冻为 True。"""
    cfg = load_config()
    mc = cfg["model"]
    model = TransferModel(
        encoder_type="tcn", n_features=12, n_target=10,
        channels=mc["tcn"]["channels"], kernel_size=mc["tcn"]["kernel_size"],
        num_blocks=mc["tcn"]["num_blocks"], dropout=mc["tcn"]["dropout"],
        latent_dim=mc["latent_dim"])
    model.freeze_encoder(True)
    for name, p in model.encoder.named_parameters():
        assert not p.requires_grad, f"S2 编码器应冻结: {name} 仍 requires_grad"
    model.freeze_encoder(False)
    for name, p in model.encoder.named_parameters():
        assert p.requires_grad, f"S3 编码器应解冻: {name}"


def test_trajectory_split_disjoint_and_covering():
    """train/val/test 轨迹集合两两不相交且覆盖全部 (完整轨迹级)。"""
    cfg = load_config()
    s = cfg["transfer"]["split"]
    tr, va, te = split_trajectories(100, [s["train"], s["val"], s["test"]], cfg["seed"])
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == 100
    # 比例近似 15/20/65
    assert 0.10 <= len(tr) / 100 <= 0.20
    assert 0.15 <= len(va) / 100 <= 0.25
    assert len(te) / 100 >= 0.55


def test_mmd_uses_hi_bins():
    """MMD 按 HI 分箱计算 (对齐键为健康阶段)。"""
    torch.manual_seed(0)
    zS = torch.randn(100, 8)
    hiS = torch.rand(100)
    zT = torch.randn(80, 8)
    hiT = torch.rand(80)
    bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    val = mmd_by_hi_bins(zS, hiS, zT, hiT, bins)
    assert torch.isfinite(val) and val >= 0


def test_mmd_smaller_for_aligned_distributions():
    """同分布 latent 的 MMD 应显著小于偏移分布 (验证度量有效)。"""
    torch.manual_seed(0)
    z = torch.randn(200, 8)
    hi = torch.rand(200)
    z_same = z + 0.01 * torch.randn(200, 8)
    z_shifted = torch.randn(200, 8) + 5.0
    bins = [(0.0, 1.0)]
    m_same = mmd_by_hi_bins(z, hi, z_same, hi, bins)
    m_shift = mmd_by_hi_bins(z, hi, z_shifted, hi, bins)
    assert m_same < m_shift


def test_transfer_model_forward_shapes():
    """adapter(x_T 10维) → encoder → heads 输出形状正确。"""
    model = TransferModel(encoder_type="tcn", n_features=12, n_target=10, latent_dim=16)
    x = torch.randn(4, 32, 10)           # (B, L, 10)
    hi, rul, z = model(x)
    assert hi.shape == (4,) and rul.shape == (4,) and z.shape == (4, 16)


# ============ S3 删失损失 / 三组公平性 / 早停不变量 (协议 §8/§13/§14/§18) ============

def _mk(pred, true, ev, lb):
    """构造 (pred[requires_grad], true, event_observed, rul_lower_bound) 张量。"""
    t = torch.tensor
    return (t(pred, dtype=torch.float32).requires_grad_(True),
            t(true, dtype=torch.float32),
            t(ev, dtype=torch.float32),
            t(lb, dtype=torch.float32))


def test_censored_not_in_huber():
    """删失样本绝不进入 Huber 项: 只改删失行的真值标签不得改变 L_O (协议 §8A)。"""
    # 行 0 = 观测 (pred 0.5, true 0.7); 行 1 = 删失
    p1, r1, ev1, lb1 = _mk([0.5, 0.5], [0.7, 0.20], [1.0, 0.0], [0.2, 0.2])
    _l1, LO1, _c1, n_o1, n_c1 = rg._censored_rul_loss(
        p1, r1, ev1, lb1, 1.0, 1.0, 1.0, 1e-6, 1.0)
    p2, r2, ev2, lb2 = _mk([0.5, 0.5], [0.7, 0.95], [1.0, 0.0], [0.2, 0.2])
    _l2, LO2, _c2, _o2, _c2n = rg._censored_rul_loss(
        p2, r2, ev2, lb2, 1.0, 1.0, 1.0, 1e-6, 1.0)
    assert (n_o1, n_c1) == (1, 1), "观测/删失样本计数错误"
    assert float(LO1.detach()) > 0.0, "观测项应为非零 (否则本测试退化为平凡)"
    assert float(LO1.detach()) == pytest.approx(float(LO2.detach())), "删失行的真值标签影响了 Huber 项"
    # 观测项等于只用观测行单独算 Huber
    ref = rg._weighted_huber(p1[:1], r1[:1], 1.0, 1.0, 1.0, 1e-6)
    assert float(LO1.detach()) == pytest.approx(float(ref.detach()), rel=1e-9)


def test_censor_hinge_zero_above_lower_bound():
    """pred >= rul_lower_bound 时删失损失恰为 0 (one-sided, 协议 §8B)。"""
    p, r, ev, lb = _mk([0.8, 0.9], [0.0, 0.0], [0.0, 0.0], [0.3, 0.9])
    loss, LO, LC, n_o, n_c = rg._censored_rul_loss(p, r, ev, lb, 1.0, 1.0, 1.0, 1e-6, 1.0)
    assert (n_o, n_c) == (0, 2)
    assert float(LC.detach()) == pytest.approx(0.0), f"pred>=lb 时 L_C 应为 0, 实为 {float(LC)}"
    assert float(LO.detach()) == pytest.approx(0.0), "无观测样本时 L_O 应为 0"
    assert float(loss.detach()) == pytest.approx(0.0)


def test_censor_hinge_positive_below_lower_bound():
    """pred < lower_bound 时 L_C = mean[(lb-pred)^2] > 0, 且 eta 线性缩放 (协议 §8B)。"""
    expect = float(np.mean([(0.5 - 0.1) ** 2, (0.6 - 0.2) ** 2]))
    p, r, ev, lb = _mk([0.1, 0.2], [0.0, 0.0], [0.0, 0.0], [0.5, 0.6])
    loss, _LO, LC, _o, _c = rg._censored_rul_loss(p, r, ev, lb, 1.0, 1.0, 1.0, 1e-6, 1.0)
    assert float(LC.detach()) > 0.0
    assert float(LC.detach()) == pytest.approx(expect, rel=1e-6)
    assert float(loss.detach()) == pytest.approx(expect, rel=1e-6), "eta=1 时 loss 应等于 L_C"
    p2, r2, ev2, lb2 = _mk([0.1, 0.2], [0.0, 0.0], [0.0, 0.0], [0.5, 0.6])
    loss2, _a, _b, _c2, _d2 = rg._censored_rul_loss(
        p2, r2, ev2, lb2, 1.0, 1.0, 1.0, 1e-6, 2.0)
    assert float(loss2.detach()) == pytest.approx(2.0 * expect, rel=1e-6), "eta 未线性作用于删失项"


def test_censor_hinge_gradient_zero_when_satisfied():
    """已满足下界的删失样本梯度必须恰为 0; 未满足者应有向上推的梯度 (协议 §8B)。"""
    p, r, ev, lb = _mk([0.9, 0.1], [0.0, 0.0], [0.0, 0.0], [0.3, 0.5])
    loss, _LO, _LC, _o, _c = rg._censored_rul_loss(p, r, ev, lb, 1.0, 1.0, 1.0, 1e-6, 1.0)
    loss.backward()
    g = p.grad.detach().numpy()
    assert g[0] == pytest.approx(0.0), f"pred(0.9)>=lb(0.3) 的样本梯度应为 0, 实为 {g[0]}"
    assert g[1] < 0.0, "pred(0.1)<lb(0.5) 的样本应有把预测推高的梯度"


def test_uncensored_path_equals_s25_weighted_huber():
    """全部 event_observed=True 时, 删失损失与 S2.5 的 _weighted_huber 数值等价。

    保证 S3 只在"确有删失"时改变损失, 未截断轨迹的训练语义与 S2.5 逐行一致。
    """
    torch.manual_seed(0)
    p = torch.rand(4, 8, requires_grad=True)
    r = torch.rand(4, 8)
    ev, lb = torch.ones(4, 8), torch.zeros(4, 8)
    loss, _LO, LC, n_o, n_c = rg._censored_rul_loss(p, r, ev, lb, 1.0, 0.1, 0.1, 1e-6, 1.0)
    ref = rg._weighted_huber(p, r, 1.0, 0.1, 0.1, 1e-6)
    assert (n_o, n_c) == (32, 0) and float(LC.detach()) == 0.0
    assert float(loss.detach()) == pytest.approx(float(ref.detach()), rel=1e-9)


def test_three_groups_same_target_ids():
    """协议 §13: 三组共用 prepare_s3 产出的同一份目标数据, 组名只影响冻结/预训练加载。"""
    src = inspect.getsource(s3g.train_group)
    assert 'data["mk_train_loader"]()' in src, "训练必须走同一份 mk_train_loader"
    assert 'data["lva"]' in src and 'data["lte"]' in src
    for bad in ("split_trajectories", "select_train_trajectories",
                "load_target", "truncate_hi=", "TargetSeqDataset"):
        assert bad not in src, f"train_group 不得自行 {bad} (会破坏三组同数据)"
    assert set(s3g.GROUPS) == {"target_only", "source_finetune", "source_mmd_finetune"}, \
        "S3 只允许这三组 (禁止新增第四个迁移方法)"


def test_three_groups_same_batch_order():
    """协议 §13: shuffle 种子由 seed 确定性推导并绑定 generator, 三组 batch order 相同。"""
    psrc = inspect.getsource(s3g.prepare_s3)
    assert "shuffle_seed = int(seed) * 1000 + 3" in psrc, "shuffle 种子必须由 seed 确定性推导"
    assert "g.manual_seed(shuffle_seed)" in psrc, "target train loader 必须绑定固定 generator"
    assert "batch_order_hash" in psrc, "必须记录 target_batch_order_hash 指纹"
    # 三组各自不得再造 target 训练乱序 (只有源域 MMD loader 允许另建 generator)
    tsrc = inspect.getsource(s3g.train_group)
    head = tsrc.split("lS = DataLoader")[0]
    assert "shuffle=True" not in head, "train_group 不得自行构造乱序 target loader"
    assert "DataLoader" not in head, "target loader 只能来自 prepare_s3"


def test_s3_earlystop_uses_val_info():
    """S3 早停沿用 candidate D 的 val info 轨迹宏平均 RMSE (协议 §14)。"""
    cfg = load_config()
    assert cfg["training"]["early_stop_metric"] == "info_macro_rmse"
    th = rg.training_hyper(cfg)
    assert th["early_stop_metric"] == "info_macro_rmse"
    src = inspect.getsource(s3g.train_group)
    assert 'early_stop_metric=th["early_stop_metric"]' in src, "早停指标必须走 training_hyper"
    assert 'c["info"]["macro"]["rmse"]' in src, "主指标必须取 info 轨迹宏平均 RMSE"
    # early_stop_value 对任何含 test 的指标名必须拒绝
    with pytest.raises(Exception):
        rg.early_stop_value({"calibers": {}}, "test_info_macro_rmse")


def test_s3_earlystop_never_reads_test():
    """结构性保证: 早停签名里没有 test loader; test 只在 best checkpoint 定下后被评估。"""
    params = list(inspect.signature(rg._train_with_early_stop).parameters)
    assert not [p for p in params if "lte" in p], f"早停签名出现 test loader: {params}"
    esrc = inspect.getsource(rg._train_with_early_stop)
    assert "lte" not in esrc, "早停函数体内出现 test loader 变量名"
    tsrc = inspect.getsource(s3g.train_group)
    assert tsrc.count('data["lva"]') == tsrc.count("_train_with_early_stop("), \
        "每次早停训练都必须且只能拿 val loader"
    head = tsrc.split("m = eval_test(")[0]
    assert 'data["lte"]' not in head, "best checkpoint 确定之前不得触碰 test loader"
    # audit / prepare 阶段也不得读 test 预测
    for fn in (s3g.prepare_s3, s3g.audit_truncation):
        s = inspect.getsource(fn)
        assert "eval_test" not in s and "collect_pred" not in s, \
            f"{fn.__name__} 不得评估任何模型输出"


def test_s25_training_parameters_preserved():
    """S3 不得改动 S2.5 冻结的 candidate D、两段降权与迁移超参 (协议 §3)。"""
    cfg = load_config()
    t = cfg["training"]
    assert cfg["s25"]["selected_candidate"] == "D"
    assert t["early_stop_metric"] == "info_macro_rmse"
    assert int(t["max_epochs"]) == 8
    assert int(t["early_stop_patience"]) == 2
    assert float(t["weight_decay"]) == pytest.approx(1.0e-3)
    assert float(t["post_eol_weight"]) == 0.1
    assert float(t["capped_weight"]) == 0.1
    assert float(cfg["experiments"]["rul_cap_ratio"]) == 0.35
    assert float(cfg["source"]["rul_cap_ratio"]) == 0.35
    assert float(cfg["transfer"]["mmd_lambda"]) == 1.0
    assert float(cfg["transfer"]["finetune_lr"]) == pytest.approx(1.0e-4)
    sc = cfg["transfer"]["split"]
    assert (sc["train"], sc["val"], sc["test"]) == (0.15, 0.20, 0.65)
    assert sc["stratify_by_eol"] is True and int(sc["n_eol_strata"]) == 4


def test_s3_frozen_config_matches_protocol():
    """协议 §5 冻结值与 config 一致, 且 S3 gate seeds 与 S2.5 全部 seed 不相交。"""
    cfg = load_config()
    tc = cfg["transfer"]
    assert float(tc["target_truncate_hi"]) == 0.55
    assert bool(tc["target_truncate_train_only"]) is True
    assert int(tc["s3_gate_train_n_traj"]) == 5
    assert float(tc["censor_hinge_eta"]) == 1.0
    assert list(tc["s3_gate_seeds"]) == [62, 63, 64]
    assert str(tc["s3_selection_rule"]) == "sha256_seed_tid"
    assert int(cfg["experiments"]["bootstrap_samples"]) >= 10000
    base = int(cfg["seed"])
    s25_seeds = ({base + o for o in cfg["s25"]["tuning_seed_offsets"]} |
                 {base + o for o in cfg["s25"]["gate_seed_offsets"]})
    assert not (set(tc["s3_gate_seeds"]) & s25_seeds), \
        f"S3 gate seeds 与 S2.5 seeds {sorted(s25_seeds)} 重叠"


def test_s3_protocol_frozen_before_results():
    """协议文件必须存在并记录全部冻结项与三种判词 (协议 §4)。"""
    proto = ROOT / "docs" / "diagnostics_s3_protocol.md"
    assert proto.exists(), "S3 协议必须先于任何结果存在"
    txt = proto.read_text(encoding="utf-8")
    for key in ("target_truncate_hi", "censor_hinge_eta", "s3_gate_seeds",
                "rul_lower_bound", "S3_TRANSFER_SIGNAL", "S3_NO_TRANSFER_SIGNAL",
                "S3_INVALID", "DESCRIPTIVE_ONLY"):
        assert key in txt, f"协议缺少 {key} 的冻结记录"
