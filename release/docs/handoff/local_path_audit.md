# Local Path & Secrets Audit — B8-PREP

## Secret / Private File Candidates

✅ No secret/private key files found.

## Absolute Path References in Source/Docs

Found 26 files with absolute path references:

| Path | Pattern | Reason | Count |
|---|---|---|---:|
| .claude/settings.json | `miniconda` | Miniconda absolute path | 22 |
| .claude/settings.local.json | `/Users/` | macOS home | 6 |
| checkpoints/basilisk_b12/electrical_provenance.json | `C:\\` | Windows C: drive | 3 |
| docs/basilisk_b5/REPRODUCE.md | `miniconda` | Miniconda absolute path | 3 |
| checkpoints/basilisk_v1/install_diag.json | `C:\\` | Windows C: drive | 2 |
| docs/basilisk_b1/REPRODUCE.md | `miniconda` | Miniconda absolute path | 2 |
| docs/basilisk_b13/REPRODUCE.md | `miniconda` | Miniconda absolute path | 2 |
| docs/basilisk_b17/REPRODUCE.md | `miniconda` | Miniconda absolute path | 2 |
| docs/basilisk_b19/REPRODUCE.md | `miniconda` | Miniconda absolute path | 2 |
| docs/basilisk_v1/installation_report.md | `C:\\` | Windows C: drive | 2 |
| docs/开发推进计划/progress.md | `C:\\` | Windows C: drive | 2 |
| t.py | `miniconda` | Miniconda absolute path | 1 |
| docs/diagnostics_20260806.md | `C:\\` | Windows C: drive | 1 |
| checkpoints/basilisk_b16/wheel_inventory.json | `miniconda` | Miniconda absolute path | 1 |
| checkpoints/basilisk_b17/thermal_parameter_registry.json | `miniconda` | Miniconda absolute path | 1 |
| data/features/dyntransfer/phm2012_v1/audit.json | `/Users/` | macOS home | 1 |
| docs/basilisk_b1/protocol.md | `miniconda` | Miniconda absolute path | 1 |
| docs/basilisk_b11/REPRODUCE.md | `miniconda` | Miniconda absolute path | 1 |
| docs/basilisk_b12/REPRODUCE.md | `C:\\` | Windows C: drive | 1 |
| docs/basilisk_b14/REPRODUCE.md | `C:\\` | Windows C: drive | 1 |
| docs/basilisk_b15/REPRODUCE.md | `C:\\` | Windows C: drive | 1 |
| docs/basilisk_b16/REPRODUCE.md | `/Users/` | macOS home | 1 |
| docs/basilisk_b17/selected_wheel_audit.md | `miniconda` | Miniconda absolute path | 1 |
| docs/basilisk_b18/REPRODUCE.md | `/Users/` | macOS home | 1 |
| docs/basilisk_v1/REPRODUCE.md | `miniconda` | Miniconda absolute path | 1 |
| scripts/handoff/local_path_audit.py | `C:\\` | Windows C: drive | 1 |

## Recommendations

- Secret/private key files: DO NOT include in Docker handoff package.
- Absolute path references: Convert to project-relative paths where possible.
- Historical frozen documents containing absolute paths: DO NOT modify original;
  ensure runtime code does not depend on them.
