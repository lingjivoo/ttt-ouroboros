# Release validation

Validation performed on 2026-09-29.

## Static and CPU checks

- all Python sources compiled;
- manifest and suite commands expanded without unresolved variables;
- four release-workflow tests passed;
- tensor-mechanics tests are executed by CI after CPU PyTorch is installed;
- Ruff passed on the package, tests and release utilities;
- the source distribution built as a wheel;
- all submitted vector figures regenerated.

## GPU integration smoke

The canonical three-arm smoke test ran with the released 125M checkpoint on an
NVIDIA H20 under PyTorch 2.9.0+cu128. It used physical book row 2, seed 42, 16
chunks and branch-only probes at positions 8 and 16.

| Arm | Status | Probe NLLs | Final drift |
|---|---|---|---:|
| Writes Off | passed | 3.199708, 3.297248 | 0.00455 |
| Closed Loop | passed | 3.199708, 3.377452 | 0.00609 |
| Fixed Generation | passed | 3.199708, 3.363771 | 0.00570 |

The identical first-probe value checks the common prefill state. These smoke
values only validate execution and must not be reported as experimental results.
