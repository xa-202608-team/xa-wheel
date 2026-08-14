# final_cleanup_log.md —— B8-HANDOFF-FINAL §10 release 目录整理

`DATA_TREE_PROTECTED = True`

## 删除前五条判据（每项均已满足）

1. **not under data/** —— 收集阶段直接跳过 `data/`；删除前另有
   `guard()` 逐条校验，命中即抛 `B8_DATA_PROTECTION_VIOLATION`
   并中止**全部**删除。本次 violations = 0。
2. **not dependency closure** —— 只删 `__pycache__` / `.pyc` /
   `.pytest_cache` 等纯缓存，以及 `checkpoints/_smoke_b5`
   （`--fast` 的临时输出，每次运行自动重建）。
3. **not referenced by REPRODUCE** —— `docs/REPRODUCE.md` 不引用任何
   缓存路径。
4. **not referenced by tests** —— 缓存目录由 Python 自动重建；
   清理后 release 声明范围测试 488 passed / 0 failed。
5. **not referenced by run scripts** —— `scripts/run_all.py --fast`
   清理后复跑，三个数字仍在 1e-6 容差内。

## 汇总

- 删除目录: **38**
- 删除文件: **0**
- 释放空间: **11.11 MB**
- 落在 `data/` 的删除项: **0**

## 明细

| 路径 | 类型 | 字节 |
|------|------|------|
| `.pytest_cache` | dir | 116,067 |
| `checkpoints/_smoke_b5` | dir | 6,338,647 |
| `scripts/__pycache__` | dir | 103,751 |
| `scripts/basilisk_b1/__pycache__` | dir | 27,915 |
| `scripts/basilisk_b11/__pycache__` | dir | 110,464 |
| `scripts/basilisk_b2/__pycache__` | dir | 68,140 |
| `scripts/basilisk_b21/__pycache__` | dir | 51,551 |
| `scripts/basilisk_b3x/__pycache__` | dir | 47,721 |
| `scripts/basilisk_b4x/__pycache__` | dir | 73,923 |
| `scripts/basilisk_b5/__pycache__` | dir | 39,428 |
| `src/__pycache__` | dir | 185 |
| `src/baselines/__pycache__` | dir | 78,891 |
| `src/data/__pycache__` | dir | 190 |
| `src/data/preprocess/__pycache__` | dir | 41,196 |
| `src/experiments/__pycache__` | dir | 93,243 |
| `src/models/__pycache__` | dir | 19,325 |
| `src/sim/__pycache__` | dir | 106,855 |
| `src/transfer/__pycache__` | dir | 114,979 |
| `src/utils/__pycache__` | dir | 6,478 |
| `tests/__pycache__` | dir | 605,139 |
| `tests/basilisk/__pycache__` | dir | 143,868 |
| `tests/basilisk_b1/__pycache__` | dir | 177,615 |
| `tests/basilisk_b11/__pycache__` | dir | 203,551 |
| `tests/basilisk_b12/__pycache__` | dir | 173,237 |
| `tests/basilisk_b13/__pycache__` | dir | 135,659 |
| `tests/basilisk_b14/__pycache__` | dir | 246,458 |
| `tests/basilisk_b15/__pycache__` | dir | 113,921 |
| `tests/basilisk_b16/__pycache__` | dir | 107,909 |
| `tests/basilisk_b17/__pycache__` | dir | 108,553 |
| `tests/basilisk_b18/__pycache__` | dir | 235,673 |
| `tests/basilisk_b19/__pycache__` | dir | 146,637 |
| `tests/basilisk_b2/__pycache__` | dir | 160,701 |
| `tests/basilisk_b21/__pycache__` | dir | 137,728 |
| `tests/basilisk_b3x/__pycache__` | dir | 83,677 |
| `tests/basilisk_b4x/__pycache__` | dir | 121,128 |
| `tests/basilisk_b5/__pycache__` | dir | 278,061 |
| `tests/basilisk_b6/__pycache__` | dir | 384,722 |
| `tests/basilisk_b7/__pycache__` | dir | 108,838 |
