"""防泄漏测试 — 同一轴承 ID 不同时出现在 train 与 val + HI 路径不含真值/病态参数。

P1 验收第 4 条 (plan §四 Phase 1)。训练严谨性: 按机台个体划分 (CLAUDE.md §7)。
依赖 data/features/wheel/schema_v1/source_features.h5 (由 wheel_features.py 生成); 未生成则 skip。

S2' 改动 5 追加: HI 计算路径的源码级约束 —— 只允许依赖 Kt_hat (实测相对偏差
mean 0.0132 / max 0.0934), 禁止 Tc_hat / b0_hat 进入。原因: T_c 与 b_0·ω 因
sgn(ω_cmd)≡1 与 ω_cmd (常值+2%正弦) 近似共线, 二者的**和**可辨识而**拆分**病态,
把病态估计量喂进 HI 会把辨识误差直接注入监督标签。
"""
import inspect
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES_H5 = ROOT / "data" / "features" / "wheel" / "schema_v1" / "source_features.h5"

# HI 计算路径中禁止出现的标识符 (真值参数 + 病态估计量)
FORBIDDEN_IN_HI = ["Tc_hat", "b0_hat", "b_true", 'params["Kt"]', "params['Kt']",
                   'params["Tc"]', "params['Tc']", 'params["b0"]', "params['b0']",
                   'params["omega0"]', "params['omega0']",
                   'attrs["Kt"]', 'attrs["Tc"]', 'attrs["b0"]', 'attrs["omega0"]']


def _load_split():
    if not FEATURES_H5.exists():
        import pytest
        pytest.skip(f"源域特征未生成: {FEATURES_H5} (先跑 wheel_features.py --synthetic)")
    import h5py
    with h5py.File(FEATURES_H5, "r") as f:
        return pd.DataFrame({
            "bearing_id": [s.decode() for s in f["bearing_id"][:]],
            "split": [s.decode() for s in f["split"][:]],
        })


def test_no_bearing_id_overlap():
    """train 与 val 的轴承 ID 集合必须不相交。"""
    df = _load_split()
    train_ids = set(df.loc[df["split"] == "train", "bearing_id"])
    val_ids = set(df.loc[df["split"] == "val", "bearing_id"])
    overlap = train_ids & val_ids
    assert not overlap, f"泄漏! 轴承同时出现在 train/val: {overlap}"
    assert train_ids, "train 轴承集合为空"
    assert val_ids, "val 轴承集合为空"


def test_split_coverage():
    """所有样本必须被划分为 train 或 val。"""
    df = _load_split()
    assert df["split"].isin(["train", "val"]).all(), "存在未划分样本"


# ================= S2' 改动 5: HI 计算路径的参数约束 =================
def _strip_comments_and_docstring(src: str) -> str:
    """去掉 # 注释与三引号文档串, 只留可执行代码 (注释里提"禁止 Tc_hat"不算违规)。"""
    import io
    import tokenize
    out = []
    prev_type = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        # 独立成句的字符串常量 = 文档串 (前一个有效 token 是 INDENT/NEWLINE/NL 或行首)
        if tok.type == tokenize.STRING and prev_type in (
                None, tokenize.INDENT, tokenize.NEWLINE, tokenize.NL):
            prev_type = tok.type
            continue
        out.append(tok.string)
        if tok.type not in (tokenize.NL,):
            prev_type = tok.type
    return "\n".join(out)


def _hi_source_text() -> str:
    """HI 计算路径的**可执行**源码 = build_hi_Tf 全文 + build_features 的 Tf 分支段。

    S5 改动 3 之后 build_hi_Tf 的调用被提到分支外 (它的 Tf_hat 还要给趋势特征用),
    故切片起点改为该调用行, 终点是 "b" 回退分支的 else —— 仍然**只**覆盖 Tf 口径下
    真正参与 HI_B 计算的语句, 不把 b 回退分支算进来。
    """
    import textwrap
    from src.sim import build_hi
    txt = inspect.getsource(build_hi.build_hi_Tf)
    bf = inspect.getsource(build_hi.build_features)
    lo = bf.index("HI_Tf, Tf_hat, Tf_fail = build_hi_Tf(")
    hi = bf.index("    else:", lo)          # b 回退分支起点
    branch = textwrap.dedent(bf[lo:hi])
    return _strip_comments_and_docstring(txt) + "\n" + _strip_comments_and_docstring(branch)


def test_hi_path_has_no_truth_or_illposed_params():
    """HI 计算路径不得出现 Tc_hat / b0_hat / 任何真值参数。

    Tf 口径只允许 Kt_hat: HI = cummax(clip((Kt_hat·mean(I_m) − mean(T_cmd) − Tf_0)
    / (Kt_hat·Im_rated − mean(T_cmd) − Tf_0), 0, 1))。
    """
    src = _hi_source_text()
    hits = [t for t in FORBIDDEN_IN_HI if t in src]
    assert not hits, f"HI 计算路径出现禁用参数 {hits}"
    assert "Kt_hat" in src or "Kt" in src, "HI 路径应显式使用 Kt_hat"


def test_trend_features_path_has_no_truth_params():
    """S5 改动 3 的趋势特征路径同样不得触碰真值 (自校准纪律对新增列一视同仁)。

    Tf_ratio 的基线 Tf_0 必须来自**序列自身**的健康段, 而不是仿真参数; 若谁把
    params["b0"] 之类拿来做基线, 特征就变成偷看真值, 这里立刻失败。
    """
    from src.sim import build_hi
    src = _strip_comments_and_docstring(inspect.getsource(build_hi.build_trend_features))
    hits = [t for t in FORBIDDEN_IN_HI if t in src]
    assert not hits, f"趋势特征路径出现禁用参数 {hits}"
    params = list(inspect.signature(build_hi.build_trend_features).parameters)
    assert params == ["Tf_hat", "sim_cfg"], \
        f"build_trend_features 签名应为 (Tf_hat, sim_cfg), 实际 {params}"


def test_build_hi_Tf_signature_excludes_illposed():
    """build_hi_Tf 的形参里不得有 Tc_hat / b0_hat 入口 (接口层面封死)。"""
    from src.sim.build_hi import build_hi_Tf
    params = list(inspect.signature(build_hi_Tf).parameters)
    assert params == ["Im", "tcmd", "Kt_hat", "sim_cfg"], \
        f"build_hi_Tf 签名应为 (Im, tcmd, Kt_hat, sim_cfg), 实际 {params}"
    for bad in ("Tc_hat", "b0_hat"):
        assert bad not in params, f"build_hi_Tf 不应接收 {bad}"


def test_hi_Tf_numerically_independent_of_illposed_params():
    """数值验证: build_hi_Tf 只接收 Kt_hat, 无从访问 Tc_hat/b0_hat (无隐式全局读取)。

    构造合成输入直接调用, 若函数内部偷读了模块级/全局的真值, 这里会因缺变量而报错。
    """
    import numpy as np
    from src.sim.build_hi import build_hi_Tf
    from src.utils import load_config
    sim = load_config()["sim"]
    n = 500
    Im = np.linspace(0.5, 2.0, n)          # 单调上升穿过 Im_rated
    tcmd = np.full(n, 0.01)
    HI, Tf_hat, Tf_fail = build_hi_Tf(Im, tcmd, 0.025, sim)
    assert HI.shape == (n,) and np.all(np.isfinite(HI))
    assert HI.min() >= 0.0 and HI.max() <= 1.0
    assert np.all(np.diff(HI) >= -1e-12), "HI 应单调非减"
    assert HI[-1] >= 1.0 - 1e-3, "I_m 已超额定, HI 末值应达 1.0"


# ================= S3 目标域 train 截断 (协议 §9/§12) =================
# 这些测试钉死的是"截断只作用于 train、train 看不到真实 EOL、选择确定性",
# 不是数字好坏。任何一条失败 → S3 判词强制 S3_INVALID。
S3_TRUNC_TOL = 1e-6


def _s3_ctx():
    """构造 S3 截断上下文 (真实目标域 h5)。缺数据则 skip。"""
    import pytest
    import h5py
    import numpy as np
    from src.utils import load_config
    from src.transfer.train_transfer import (
        split_trajectories_from_cfg, load_target, select_train_trajectories)
    cfg = load_config(str(ROOT / "configs" / "wheel.yaml"))
    tc = cfg["transfer"]
    h5p = ROOT / tc["target_feature_path"]
    if not h5p.exists():
        pytest.skip(f"缺目标域特征 {h5p}")
    seed = int(tc["s3_gate_seeds"][0])
    with h5py.File(h5p, "r") as f:
        n_obs = sum(1 for k in sorted(f.keys())
                    if bool(f[k].attrs.get("event_observed",
                                           np.asarray(f[k].get("label_fail", [])).any())))
    tr, va, te, _ = split_trajectories_from_cfg(h5p, n_obs, tc, seed, observed_only=True)
    tr = [int(t) for t in tr]
    sel, sel_hash = select_train_trajectories(
        seed, tr, int(tc["s3_gate_train_n_traj"]), rule=str(tc["s3_selection_rule"]))
    cen = {}
    load_target(h5p, False, observed_only=True,
                truncate_hi=float(tc["target_truncate_hi"]), truncate_tids=sel,
                split_role="train", cap_ratio=float(cfg["source"]["rul_cap_ratio"]),
                censor_out=cen)
    audit = {int(a["tid"]): a for a in cen["audit"]}
    return {"cfg": cfg, "tc": tc, "h5": h5p, "seed": seed, "tr": tr,
            "sel": sel, "sel_hash": sel_hash, "va": [int(t) for t in va],
            "te": [int(t) for t in te], "audit": audit, "cen": cen}


def test_s3_train_has_no_failure_label():
    """被选中的 S3 train 轨迹截断后 label_fail 合计必须为 0, event_observed 必须为 0。"""
    c = _s3_ctx()
    for t in c["sel"]:
        a = c["audit"][t]
        assert a["truncated"], f"tid {t} 未被截断"
        assert a["failure_label_count"] == 0, f"tid {t} 截断段仍含 label_fail=1 (泄漏真实 EOL)"
        assert a["event_observed"] == 0, f"tid {t} event_observed 应为 0 (右删失)"


def test_s3_train_hi_not_above_truncate():
    """截断后保留段 max(HI) <= truncate_hi + tol, 其后的真实 EOL 不可见。"""
    c = _s3_ctx()
    thr = float(c["tc"]["target_truncate_hi"])
    for t in c["sel"]:
        a = c["audit"][t]
        assert a["max_hi"] <= thr + S3_TRUNC_TOL, \
            f"tid {t} 截断后 max_hi={a['max_hi']} > {thr}"
        assert a["observed_n"] < a["original_n"], f"tid {t} 长度未变短, 截断无效"


def test_s3_val_not_truncated():
    """val 轨迹绝不截断 (协议 §6 绝对禁止)。"""
    c = _s3_ctx()
    for t in c["va"]:
        a = c["audit"][t]
        assert not a["truncated"], f"val tid {t} 被截断"
        assert a["observed_n"] == a["original_n"], f"val tid {t} 长度被改动"
        assert a["event_observed"] == 1, f"val tid {t} event_observed 被改为 0"


def test_s3_test_not_truncated():
    """test 轨迹绝不截断 (协议 §6 绝对禁止)。"""
    c = _s3_ctx()
    for t in c["te"]:
        a = c["audit"][t]
        assert not a["truncated"], f"test tid {t} 被截断"
        assert a["observed_n"] == a["original_n"], f"test tid {t} 长度被改动"
        assert a["event_observed"] == 1, f"test tid {t} event_observed 被改为 0"


def test_s3_truncate_rejected_for_non_train_role():
    """接口层面封死: split_role != 'train' 时传 truncate_hi 直接抛错。"""
    import pytest
    from src.transfer.train_transfer import load_target
    c = _s3_ctx()
    for role in ("val", "test", "full"):
        with pytest.raises(ValueError, match="截断只允许作用于 train"):
            load_target(c["h5"], False, observed_only=True, truncate_hi=0.55,
                        truncate_tids=c["sel"], split_role=role, cap_ratio=0.35)


def test_s3_train_val_test_disjoint():
    """S3 使用的 train 子集与 val/test 两两不相交, 且是原 train 划分的子集。"""
    c = _s3_ctx()
    S, V, T = set(c["sel"]), set(c["va"]), set(c["te"])
    assert S <= set(c["tr"]), "S3 选中的轨迹必须来自原 train 划分"
    assert not (S & V) and not (S & T) and not (V & T), "train/val/test 相交"
    assert len(S) == int(c["tc"]["s3_gate_train_n_traj"])


def test_s3_selection_deterministic():
    """确定性选择 (协议 §12): 同 seed 重复调用结果与 hash 逐字一致, 不看 val/test。"""
    import inspect
    from src.transfer.train_transfer import select_train_trajectories
    c = _s3_ctx()
    n = int(c["tc"]["s3_gate_train_n_traj"])
    rule = str(c["tc"]["s3_selection_rule"])
    for _ in range(3):
        sel2, h2 = select_train_trajectories(c["seed"], c["tr"], n, rule=rule)
        assert sel2 == c["sel"] and h2 == c["sel_hash"], "同 seed 选择结果不稳定"
    # 传入顺序被打乱也必须给出同一结果 (只依赖 seed 与 tid 集合)
    sel3, h3 = select_train_trajectories(c["seed"], list(reversed(c["tr"])), n, rule=rule)
    assert sel3 == c["sel"] and h3 == c["sel_hash"], "选择结果依赖了输入顺序"
    # 选择函数**引用的标识符**里不得出现 val/test/指标相关名字
    # (只看 Name/Attribute/关键字参数名 —— 排除 docstring、# 注释与 ValueError 之类的内建异常)
    import ast
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(select_train_trajectories))).body[0]
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    names -= {"ValueError", "TypeError", "KeyError"}          # 内建异常名不算触碰
    for bad in ("val", "test", "rmse", "loss", "eval"):
        hit = [n for n in names if bad in n.lower()]
        assert not hit, f"确定性选择疑似触碰 {bad}: {hit}"


def test_s3_lower_bound_never_uses_future_eol():
    """rul_lower_bound 必须只由观测窗口右端决定, 与真实 EOL 无关 (协议 §10)。

    数值验证: 对每条截断轨迹, lb 的最大值必须 == min(t_obs_end, cap_ratio*n_full),
    即 t=0 处的下界; 且 lb 严格递减到 0 —— 若实现偷读了 rul (真值), 最大值会等于
    真实 EOL 而不是 t_obs_end (二者在本数据上差 1 个数量级)。
    """
    import numpy as np
    c = _s3_ctx()
    cap = float(c["cfg"]["source"]["rul_cap_ratio"])
    lb = np.asarray(c["cen"]["rul_lower_bound"], dtype=float)
    ro = np.asarray(c["cen"]["rul_observed"], dtype=float)
    ev = np.asarray(c["cen"]["event_observed"], dtype=bool)
    for t in c["sel"]:
        a = c["audit"][t]
        expect_max = min(float(a["t_obs_end"]), cap * float(a["original_n"]))
        assert abs(a["rul_lower_bound_max"] - expect_max) < 1e-6, \
            f"tid {t} lb 最大值 {a['rul_lower_bound_max']} != t_obs_end 派生值 {expect_max}"
        assert a["rul_lower_bound_min"] == 0.0
        # 真实 EOL 必须显著大于 t_obs_end (截断确实丢掉了失效信息)
        assert a["t_obs_end"] < a["original_n"] - 1
    # 删失行的真实 RUL 一律不可见
    assert np.all(np.isnan(ro[~ev])), "删失行的真实 RUL 必须被置为 NaN (不可见)"
    assert np.all(np.isfinite(lb)), "lower bound 必须全部有限"
