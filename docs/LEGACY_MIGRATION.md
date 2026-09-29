# Legacy migration audit

The research workspace contained many exploratory scripts and scheduler queues.
This release keeps only entry points tied to a documented paper protocol. The
following files were intentionally excluded because they were superseded, were
site-specific queue controllers, or lacked a frozen protocol. Their omission
does not delete the private archival snapshot.

## Excluded exploratory scripts

- `a5_flips.py`
- `agent_reeval.py`
- `agent_stats.py`
- `alf_tasktypes.py`
- `alfworld_demos.py`
- `alfworld_envcheck.py`
- `analyze_exposure_density.py`
- `analyze_reviewer_focused.py`
- `analyze_reviewer_followups.py`
- `analyze_reviewer_priority.py`
- `analyze_settlement_value.py`
- `analyze_update_strength_sweep.py`
- `anchor_causal.py`
- `attt_closed_loop.py`
- `bench_step1.py`
- `book_blocked.py`
- `cache_prior.py`
- `contam.py`
- `convert_jax_ckpt.py`
- `delta_diag.py`
- `determ_check.py`
- `dist_match.py`
- `dose_check.py`
- `dose_verdict.py`
- `dump_jax_ckpt.py`
- `eval_val.py`
- `fig_fourarm.py`
- `flip_overlap.py`
- `icl.py`
- `initial_nll_audit.py`
- `jax_forward_dump.py`
- `mechanism_e4.py`
- `needle.py`
- `npy_to_zarr.py`
- `one_b_preflight.py`
- `order_inv.py`
- `oss_probes.py`
- `pareto.py`
- `phase_stats.py`
- `probes.py`
- `probes2.py`
- `pt_forward_compare.py`
- `qwen_passkey_128k.py`
- `qwen_prepare_pg19.py`
- `qwen_reviewer_numeric_audit.py`
- `qwen_reviewer_p0a.py`
- `qwen_reviewer_p0c.py`
- `remedies.py`
- `reviewer_3b_replay.py`
- `reviewer_focused_numeric_audit.py`
- `settlement_value_pilot.py`
- `spiral.py`
- `sw_arms.py`
- `sw_env.py`
- `sw_probe.py`
- `sw_reeval.py`
- `sw_repro.py`
- `task_stats.py`
- `tokenize_dclm.py`
- `ttt_task.py`
- `twc_calibrate.py`
- `twc_demos.py`
- `twc_env.py`
- `whichbooks.py`
- `write_timing.py`
- `ws_arms.py`
- `ws_build_catalogue.py`
- `ws_canaries.py`
- `ws_env.py`
- `ws_probe.py`
- `ws_slice_check.py`

## Excluded scheduler state

All Python queue daemons, occupancy workers, generated `.sbatch` files, retry
manifests, logs, partial downloads, and host-specific paths were excluded. Two
generic Slurm templates are provided under `cluster/slurm/`.
