# SAFE_DELETE Execution Plan — B8-PREP

## 🔒 Guard Assertions (Passed = May Execute)

| Guard | Status |
|---|---|
| No path startswith `data/` | ✅ PASS |
| No path startswith `checkpoints/` | ✅ PASS |
| No path startswith `docs/basilisk_` | ✅ PASS |
| No `*.h5` files slated | ✅ PASS |
| No `*.pt` / `*.pth` / `*.ckpt` files slated | ✅ PASS |
| No `*.json` metrics files slated | ✅ PASS |
| No config YAML files slated | ✅ PASS |
| No test source files slated | ✅ PASS |
| No `src/` source files slated | ✅ PASS |

---

## Files to Delete

All files below are categorized **SAFE_DELETE** per §2 definition.

| Category | Count | Total Size |
|---|---:|---:|
| Python cache (`__pycache__/`, *.pyc) | TBD | TBD |
| pytest cache | TBD | TBD |
| mypy / ruff cache | TBD | TBD |
| Jupyter checkpoints | TBD | TBD |
| Temp editor files (*.swp / *~) | TBD | TBD |
| Empty directories | TBD | TBD |

---

## Execution Constraints

**DO NOT EXECUTE:**

- `git clean -xfd` — ❌ Risk to uncommitted critical data
- `rm -rf data/` — ❌ Violates READ_ONLY_CRITICAL
- "Delete all untracked files" — ❌ Too broad risk

**ONLY EXECUTE:**

- Per-file deletion of explicitly categorized SAFE_DELETE items
- Only after inventory scan fully completes and guards are verified

---

## Post-Deletion Output

After execution, will generate:
- `docs/handoff/deletion_log.md`
- `files_deleted` count
- `bytes_reclaimed` summary
