# experiments/

`afo collect` / `afo run` write one directory per experiment here:
`<name>_<timestamp>_<confighash>/`, or the name passed with `--exp-dir`. Outputs are git-ignored.
Regenerate them with:

```bash
afo run configs/experiments/smoke.yaml          --exp-dir experiments/smoke           # ~20 s
afo run configs/experiments/main_simulated.yaml --exp-dir experiments/main-simulated  # ~2 min
afo run configs/experiments/null_control.yaml   --exp-dir experiments/null-control    # ~1 min
```

Each directory contains the following:

| Path | Contents |
|---|---|
| `config.yaml` | Resolved configuration. Rerunning it reproduces the experiment. |
| `manifest.json` | Git commit, package versions, model id/revision, task-version hash, activation layout |
| `tasks.jsonl` | The exact task instances used |
| `trajectories.jsonl` | One trajectory per line (schema: `src/afo/collection/trajectory.py`) |
| `activations/<run_id>.npy` | float16 `[n_steps, n_layers, hidden_dim]`; row t = decision point t |
| `splits.json` | Family → train/val/test assignment |
| `features/` | Step-level dataset (`dataset.npz`) and split summary |
| `monitors/` | Pickled monitors per horizon, the layer sweep, and `index.json` with thresholds |
| `eval/` | `metrics.json`, `predictions.jsonl`, `generalization.json`, and `plots/` |
| `interventions/` | Per-condition trajectories and `summary.json` |
| `report/results.md` | Generated results report |
