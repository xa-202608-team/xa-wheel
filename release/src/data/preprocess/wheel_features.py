"""src/data/preprocess/wheel_features.py

源域 (XJTU-SY) 振动 -> 12 维退化特征 + HI/RUL 标签派生。

特征 (12维, 满足 plan 要求 10<=F<=15):
  时域 8: rms, peak, kurtosis, skewness, crest, clearance, shape, std
  频域 2: spec_centroid, spec_entropy
  包络域 2: env_kurtosis (Hilbert), band_energy (高频共振带)

标签 (派生, 符合 CLAUDE.md 数据纪律 §4 — 非数据集原始字段):
  HI    = isotonic(RMS_norm) 与 isotonic(env_kurtosis_norm) 等权融合,
          端点归一到 [0,1], 失效=1 (大=退化严重)
  RUL   = min(T_life - t, R_max), R_max = rul_cap_ratio × T_life (早期封顶)

划分: 严格按轴承个体 (杜绝同一退化轨迹泄漏)。
  默认 val = {Bearing1_5, Bearing2_5, Bearing3_5} (每工况第5个), 其余 train。

用法:
  python -m src.data.preprocess.wheel_features --config configs/wheel.yaml --report
  python -m src.data.preprocess.wheel_features --config configs/wheel.yaml --synthetic --report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats as sst
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.utils import load_config  # noqa: E402

FS_DEFAULT = 25600  # XJTU-SY 采样率

FEATURE_NAMES = [
    "rms", "peak", "kurtosis", "skewness", "crest", "clearance",
    "shape", "std", "spec_centroid", "spec_entropy", "env_kurtosis", "band_energy",
]


# ---------------------------------------------------------------- 特征提取
def extract_features(x: np.ndarray, fs: int) -> np.ndarray:
    """单窗振动信号 -> 12 维退化特征向量。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # --- 时域 ---
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    kurt = sst.kurtosis(x, fisher=False)            # 正态=3, 冲击/退化升高
    skew = sst.skew(x)
    mean_abs = np.mean(np.abs(x))
    crest = peak / (rms + 1e-12)                     # 峰值因子
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(x)))
    clearance = peak / ((mean_sqrt_abs + 1e-12) ** 2)  # 裕度因子 (对早期点蚀敏感)
    shape = rms / (mean_abs + 1e-12)                 # 波形因子
    std = np.std(x)
    # --- 频域 (Welch 功率谱) ---
    nperseg = min(n, 4096)
    f, P = sps.welch(x, fs=fs, nperseg=nperseg)
    P = P + 1e-12
    spec_centroid = np.sum(f * P) / np.sum(P)        # 谱重心
    p_norm = P / np.sum(P)
    spec_entropy = -np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(P))
    # --- 包络域 (Hilbert) ---
    env = np.abs(sps.hilbert(x))
    env_kurt = sst.kurtosis(env, fisher=False)       # 轴承故障最敏感特征
    band = f > np.percentile(f, 66)                  # 高频 1/3 段能量比
    band_energy = np.sum(P[band]) / (np.sum(P) + 1e-12)
    return np.array([
        rms, peak, kurt, skew, crest, clearance, shape, std,
        spec_centroid, spec_entropy, env_kurt, band_energy,
    ], dtype=np.float64)


HORIZ_COL = "Horizontal_vibration_signals"
MIN_VALID_SAMPLES = 4096      # 单窗最少有效采样点 (低于此判为损坏窗)
MAX_CORRUPT_RATIO = 0.02      # 单轴承允许的损坏窗占比上限


def read_csv_signal(path: Path) -> np.ndarray:
    """读 XJTU-SY csv -> 水平振动通道 1D 序列。

    XJTU-SY 原始文件**带表头** (Horizontal/Vertical_vibration_signals);
    优先按列名取水平通道, 兼容无表头的旧版本文件 (退化为第 0 列)。
    个别文件存在截断/乱码行 (如 Bearing3_1/551.csv), 跳过坏行后取数值部分。
    """
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    if HORIZ_COL in df.columns:
        v = df[HORIZ_COL]
    elif str(df.columns[0]).replace(".", "", 1).replace("-", "", 1).isdigit():
        # 首行即数据 (无表头), 重读并取第 0 列
        v = pd.read_csv(path, header=None, on_bad_lines="skip", engine="python").iloc[:, 0]
    else:
        v = df.iloc[:, 0]
    x = pd.to_numeric(v, errors="coerce").to_numpy(dtype=np.float64)
    return x[np.isfinite(x)]


def _file_index(p: Path) -> int | None:
    """XJTU-SY 文件名为**采集分钟序号** (1.csv ... N.csv)。

    必须按数值排序 (字典序会给出 1,10,100,101,...,11 的错误时间顺序),
    且 t_idx 须取真实编号 — Bearing3_1 编号 1..2538 但实际只有 2079 个文件,
    用枚举下标会把 2538 min 寿命压缩成 2079, 导致 RUL 标签整体偏移。
    """
    return int(p.stem) if p.stem.isdigit() else None


def process_bearing(bearing_dir: Path, fs: int) -> pd.DataFrame:
    """一个轴承的全部 csv -> 特征时间序列。

    t_idx = 文件编号 - 1 (真实采集分钟, 0-based), 缺失编号即时间轴空洞, 保留不填补。
    """
    indexed = sorted(
        ((i, p) for p in bearing_dir.glob("*.csv") if (i := _file_index(p)) is not None)
    )
    skipped = [p.name for p in bearing_dir.glob("*.csv") if _file_index(p) is None]
    if not indexed:
        raise FileNotFoundError(f"{bearing_dir} 下无编号 csv 文件")

    rows, corrupt = [], []
    for num, p in indexed:
        try:
            x = read_csv_signal(p)
            if x.size < MIN_VALID_SAMPLES:
                raise ValueError(f"有效采样点仅 {x.size} (<{MIN_VALID_SAMPLES})")
            rows.append([num - 1] + list(extract_features(x, fs)))
        except Exception as e:
            corrupt.append(f"{p.name}: {type(e).__name__}: {e}")

    ratio = len(corrupt) / len(indexed)
    if ratio > MAX_CORRUPT_RATIO:
        raise RuntimeError(
            f"{bearing_dir.name} 损坏窗占比 {ratio:.1%} 超过 {MAX_CORRUPT_RATIO:.0%}, "
            f"前 3 例: {corrupt[:3]}"
        )
    if corrupt or skipped:
        print(f"   [{bearing_dir.name}] 跳过损坏窗 {len(corrupt)} 个"
              f"{' / 非编号文件 ' + str(len(skipped)) + ' 个' if skipped else ''}"
              f": {[c.split(':')[0] for c in corrupt][:5]}")
    return pd.DataFrame(rows, columns=["t_idx"] + FEATURE_NAMES)


# ---------------------------------------------------------------- 标签派生
def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def construct_labels(df: pd.DataFrame, rul_cap_ratio: float) -> pd.DataFrame:
    """派生 HI (单调化端点归一) + capped RUL。

    寿命 T_life 取**真实时间轴长度** t_idx.max()+1 (而非窗数 len(df)),
    否则 Bearing3_1 这类存在缺失窗的轨迹 RUL 会被系统性压缩。
    isotonic 回归同样以 t_idx 为自变量, 保证时间不均匀采样下的正确单调化。
    """
    df = df.sort_values("t_idx").reset_index(drop=True)
    t = df["t_idx"].to_numpy(dtype=np.float64)
    T_life = float(t.max()) + 1.0
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    rms_iso = iso.fit_transform(t, _minmax(df["rms"].values))
    ek_iso = iso.fit_transform(t, _minmax(df["env_kurtosis"].values))
    hi = _minmax(0.5 * rms_iso + 0.5 * ek_iso)       # 失效点 -> 1
    remaining = (T_life - 1.0) - t
    r_max = rul_cap_ratio * T_life
    rul = np.minimum(remaining, r_max)
    df["hi"] = hi
    df["rul"] = rul
    df["rul_cap"] = r_max
    df["t_life"] = T_life
    return df


def _monotonic_violation_rate(data: pd.DataFrame) -> float:
    total, viol = 0, 0
    for _, g in data.groupby("bearing_id"):
        hi = g.sort_values("t_idx")["hi"].values
        d = np.diff(hi)
        total += len(d)
        viol += int(np.sum(d < -1e-9))
    return viol / max(total, 1)


# ---------------------------------------------------------------- 合成数据 (验证逻辑)
def make_synthetic(n_bearings: int = 3, n_pts: int = 128, fs: int = FS_DEFAULT, seed: int = 0):
    """合成退化轴承 (测试): 基线正弦 + 随时间增强的周期冲击 -> 特征应单调退化。"""
    rng = np.random.default_rng(seed)
    out = {}
    t = np.arange(n_pts)
    amp_growth = 1.0 + 0.05 * t
    for b in range(n_bearings):
        csvs = []
        for i in range(n_pts):
            n_samp = 32768
            base = 0.5 * np.sin(2 * np.pi * 50 * np.arange(n_samp) / fs)
            imp = np.zeros(n_samp)
            for ix in range(0, n_samp, 400):           # 周期故障冲击
                decay = np.exp(-np.arange(min(60, n_samp - ix)) / 8.0)
                seg = decay * np.cos(2 * np.pi * 2000 * np.arange(len(decay)) / fs)
                imp[ix:ix + len(seg)] += amp_growth[i] * seg
            noise = 0.1 * rng.standard_normal(n_samp)
            csvs.append(base + imp + noise)
        out[f"Bearing{b + 1}_syn"] = csvs
    return out


# ---------------------------------------------------------------- 主流程
def run(args) -> None:
    cfg = load_config(args.config)
    fs = cfg["source"].get("sample_rate_hz", FS_DEFAULT)
    cap_ratio = cfg["source"].get("rul_cap_ratio", 0.35)
    val_ids = set(cfg["source"]["split"].get("val_bearing_ids", []) or [])
    bearing_ids = []
    all_dfs = []

    if args.synthetic:
        print(">> 合成数据模式 (验证特征工程逻辑)")
        synth = make_synthetic(seed=cfg.get("seed", 42))
        for bid, csvs in synth.items():
            rows = [[i] + list(extract_features(sig, fs)) for i, sig in enumerate(csvs)]
            df = construct_labels(pd.DataFrame(rows, columns=["t_idx"] + FEATURE_NAMES), cap_ratio)
            df["bearing_id"] = bid
            all_dfs.append(df)
            bearing_ids.append(bid)
        if not val_ids:
            val_ids = {bearing_ids[-1]}
    else:
        xjtu = ROOT / "data" / "source" / "XJTU-SY"
        if not xjtu.exists():
            print(f"!! 未找到 {xjtu}; 先 bash scripts/data/download_source.sh, 或用 --synthetic 验证逻辑")
            sys.exit(1)
        conditions = {"35Hz12kN": "1", "37.5Hz11kN": "2", "40Hz10kN": "3"}
        for cond, idx in conditions.items():
            for n in range(1, 6):
                bid = f"Bearing{idx}_{n}"
                bdir = xjtu / cond / bid
                if not bdir.is_dir():
                    print(f"   跳过 {bdir} (不存在)")
                    continue
                df = construct_labels(process_bearing(bdir, fs), cap_ratio)
                df["bearing_id"] = bid
                df["condition"] = cond
                all_dfs.append(df)
                bearing_ids.append(bid)
        if not val_ids:
            val_ids = {"Bearing1_5", "Bearing2_5", "Bearing3_5"}
            print(f">> 默认 val 轴承: {sorted(val_ids)} (写回 configs/wheel.yaml source.split.val_bearing_ids)")

    data = pd.concat(all_dfs, ignore_index=True)
    data["split"] = np.where(data["bearing_id"].isin(val_ids), "val", "train")

    feat_mat = data[FEATURE_NAMES].values
    assert 10 <= feat_mat.shape[1] <= 15, f"特征维度 {feat_mat.shape[1]} 不在 [10,15]"
    assert not np.any(np.isnan(feat_mat)) and not np.any(np.isinf(feat_mat)), "特征含 NaN/Inf"
    hi_all = data["hi"].values
    assert np.all(hi_all >= -1e-9) and np.all(hi_all <= 1 + 1e-9), \
        f"HI 越界 [{hi_all.min()}, {hi_all.max()}]"
    assert np.all(data["rul"].values >= -1e-9), "RUL 出现负值"
    missing_val = val_ids - set(bearing_ids)
    assert not missing_val, f"配置的 val 轴承不存在: {sorted(missing_val)}"
    assert (data["split"] == "train").any() and (data["split"] == "val").any(), \
        "train/val 有一侧为空"
    assert not (set(data[data["split"] == "train"]["bearing_id"])
                & set(data[data["split"] == "val"]["bearing_id"])), \
        "train/val 轴承个体重叠 — 存在退化轨迹泄漏"

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(out_dir / "source_features.h5", data)
    print(f">> 已写出: {out_dir / 'source_features.h5'}  "
          f"({len(data)} windows, {len(bearing_ids)} bearings)")

    if args.report:
        hi = data["hi"].values
        print("\n===== P1 特征工程报告 =====")
        print(f"特征矩阵 shape = {feat_mat.shape}  (F={feat_mat.shape[1]}, 满足 10<=F<=15)")
        print(f"NaN/Inf 检查   = 通过")
        print(f"HI 范围        = [{hi.min():.4f}, {hi.max():.4f}]")
        print(f"RUL cap ratio  = {cap_ratio}  (R_max = {cap_ratio * 100:.0f}% 寿命)")
        caps = data.groupby("bearing_id")["rul_cap"].first().to_dict()
        print(f"各轴承 R_max   = { {k: round(float(v), 1) for k, v in caps.items()} }")
        print(f"划分           = train {int((data['split']=='train').sum())} "
              f"/ val {int((data['split']=='val').sum())} 窗")
        print(f"train 轴承     = {sorted(set(data[data['split']=='train']['bearing_id']))}")
        print(f"val   轴承     = {sorted(set(data[data['split']=='val']['bearing_id']))}")
        print("轴承个体重叠   = 无 (按 bearing_id 划分, 无轨迹泄漏)")
        cov = data.groupby("bearing_id").agg(n_win=("t_idx", "size"), t_life=("t_life", "first"))
        gappy = cov[cov["n_win"] < cov["t_life"]]
        if len(gappy):
            print("时间轴缺窗     = " + ", ".join(
                f"{k}: {int(r.n_win)}/{int(r.t_life)} min" for k, r in gappy.iterrows()))
        else:
            print("时间轴缺窗     = 无")
        print(f"HI 单调违例率  = {_monotonic_violation_rate(data):.4f} (isotonic 保证, 应≈0)")
        print("==========================\n")


def _save(path: Path, data: pd.DataFrame) -> None:
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=data[FEATURE_NAMES].values)
        f.create_dataset("hi", data=data["hi"].values)
        f.create_dataset("rul", data=data["rul"].values)
        f.create_dataset("rul_cap", data=data["rul_cap"].values)
        f.create_dataset("t_life", data=data["t_life"].values)
        f.create_dataset("t_idx", data=data["t_idx"].values)
        f.create_dataset("feature_names",
                         data=np.array(FEATURE_NAMES, dtype="S"))
        f.create_dataset("bearing_id",
                         data=np.array(data["bearing_id"].astype(str).values, dtype="S"))
        f.create_dataset("split",
                         data=np.array(data["split"].astype(str).values, dtype="S"))
    data.to_csv(path.with_suffix(".csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wheel.yaml")
    ap.add_argument("--out", default="data/features/wheel/schema_v1")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="合成数据验证逻辑(无真实数据时)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
