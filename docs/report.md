# Can We See Agent Failures Coming? A Leakage-Resistant Framework for Predicting and Mitigating Unsafe LLM Agent Actions

*Research report, draft v0.1 (2026-11-02). Status: methodology complete, validated on a simulated
backend. The real-model study is not yet run (see §7).*

## Abstract

We present the Agent Failure Observatory (AFO), a framework for testing whether an LLM agent's unsafe
or goal-inconsistent actions can be predicted *before* they are taken, and whether such predictions
can drive interventions that help. AFO runs agents in a fully sandboxed tool environment covering
three failure categories: goal drift, prompt injection, and unsafe tool use. The 33 machine-validated
task templates expand into 456 parameterised instances.

AFO logs every decision point together with the residual-stream activations captured before the action
is generated. It compares four monitors on held-out task families: majority, behavioural, linear probe,
and combined. All comparisons use validation-only threshold selection and template-clustered bootstrap
CIs.

To validate the pipeline, we run a simulated policy whose hidden states contain risk information by
construction (positive control) or contain none (negative control).

- **Positive control:** behavioural monitoring beats the majority baseline (ΔAUROC 0.204 [0.168,
  0.243]), and adding activations improves on behaviour (ΔAUROC 0.076 [0.037, 0.126]).
- **Negative control:** the probe falls to chance (AUROC 0.497 [0.455, 0.539]) and the combined monitor
  shows no gain (ΔAUROC −0.020 [−0.042, 0.003]). This is evidence that the evaluation does not
  manufacture probe advantages through leakage.
- **Interventions:** monitor-triggered interventions cut violations by 27–32 percentage points
  (pp), against 8–13 pp for rate-matched random triggers. An approval gate pays for this with 17 pp
  lower task success, while warnings and reconsideration do not.

These results describe the simulator and not real models. They establish that the measurement
apparatus is ready for real-model experiments.

## 1. Introduction

Autonomous LLM agents fail in ways that are costly once an action has been executed: forwarding a
mailbox to an attacker, deleting audit logs, or quietly working on someone else's request. Most
evaluations score these failures after the fact. Runtime safety instead needs **prospective**
detection, meaning a signal available at the decision point, early enough to act on, that does not
block legitimate work.

We ask five research questions:

- **RQ1:** do behavioural trajectories predict failures?
- **RQ2:** do internal activations add information beyond behaviour?
- **RQ3:** how early can failures be predicted?
- **RQ4:** do monitor-triggered interventions reduce violations without excessive cost?
- **RQ5:** do monitors generalise to unseen failure categories?

**Contributions.**

1. A sandboxed, replayable agent environment with 33 templates. Each template is machine-validated:
   its reference plan succeeds, its unsafe action triggers a labelled rule, and its hazard is reachable.
   Success and safety are labelled separately.
2. Activation capture aligned to decision points, with tests guaranteeing that recorded
   representations never include the action they predict.
3. An evaluation protocol designed against leakage: family-level splits, preprocessing and thresholds
   fit on train/validation data only, cluster bootstrap CIs, and paired comparisons.
4. A **positive/negative-control** methodology using a simulated policy. It checks that the pipeline
   finds a planted signal and does not hallucinate a missing one.
5. An intervention study with rate-matched random baselines, a refuse-all reference, and post-hoc cost
   accounting for blocked legitimate actions.

## 2. Related work (to be expanded)

Agent safety benchmarks score failures after the fact on curated tasks. Prompt-injection benchmarks
study untrusted-content attacks. Linear probes have been used to read truthfulness and deception from
LLM representations, and monitor/guardrail systems gate tool calls at runtime. AFO combines these
threads: it asks whether *prospective* monitors trained on behavioural and representational features
can predict failure across categories, and it measures the safety/utility trade-off of acting on
those predictions.

## 3. Methodology

See `docs/methodology.md` for full definitions. In brief:

**Environments.** There are three categories with 11 templates each. Every template expands along
hazard strength {none, low, medium, high}, placement {early, late carrier}, and difficulty {easy:
permissions and guidance stated; hard: neither, plus decoys}. Tools are simulated: files, search, HTTP,
e-mail, shell, and submit.

**Labels.**

- A *safety violation* is an attempted action matching a violation rule. Blocked attempts count.
- *Failure onset* `k` is the first such step.
- At decision point `t`, the horizon-`h` label is `1[t ≤ k < t+h]`.
- Post-onset states are excluded.

**Monitors.** We compare a majority baseline, an L2 logistic-regression model on 20 behavioural
features, a per-layer L2 probe on decision-point activations (layer chosen on validation), and a
combined model (behavioural features plus the probe's layer). C is chosen on validation, and the
decision threshold maximises validation F1.

**Splits.** Families are assigned to train, val, or test (60/20/20), stratified by category. The test
split therefore holds only unseen task families. A leave-one-environment-out (LOEO) analysis tests
cross-category transfer.

**Interventions.** Three interventions are tested: a warning, an approval gate on side-effecting
actions, and reconsideration. Each is triggered by the combined monitor (h=3) at a validation threshold
for 10% per-step FPR, by a random trigger at the monitor's observed rate, or always (gate only).

## 4. Experimental setup

| | Positive control (`main-simulated`) | Negative control (`null-control`) |
|---|---|---|
| Policy | Simulated, `activation_signal = 1` | Simulated, `activation_signal = 0` |
| Instances × seeds | 456 × 3 = 1,368 trajectories | 456 × 3 = 1,368 |
| Activations | 8 layers × 64 dims, decision-point | same layout, nuisance only |
| Test split | 210 trajectories, 863 decision points, 5 unseen templates | identical split |
| Bootstrap | 1,000 resamples over templates | 1,000 |

Overall, the positive-control data have 35.5% safety violations (37/35/35% for goal drift, injection,
and tool safety) and 92.3% task success. Clean (`none`) instances never violate.

Reproduce with `afo run configs/experiments/main_simulated.yaml` and
`afo run configs/experiments/null_control.yaml`. Each run takes about 2 minutes on a laptop CPU.

## 5. Results

### 5.1 Failure prediction (RQ1, RQ2)

Test split, horizon 3, prevalence 0.276 (238 of 863 positive):

| Monitor | AUROC [95% CI] | AUPRC | Recall | FPR | Brier |
|---|---|---|---|---|---|
| Majority | 0.500 | 0.276 | 0.00 | 0.00 | 0.200 |
| Behavioural | 0.704 [0.668, 0.743] | 0.458 | 0.60 | 0.27 | 0.215 |
| Probe (layer 4) | 0.772 [0.745, 0.800] | 0.667 | 0.43 | 0.06 | 0.176 |
| Combined (layer 4) | 0.779 [0.745, 0.809] | 0.673 | 0.54 | 0.11 | 0.170 |

Paired differences (horizon 3):

| Comparison | Positive control | Negative control |
|---|---|---|
| Behavioural − majority | **+0.204** [0.168, 0.243] | **+0.204** [0.168, 0.243] |
| Probe − behavioural | **+0.069** [0.025, 0.118] | −0.206 [−0.252, −0.171] |
| Combined − behavioural | **+0.076** [0.037, 0.126] | −0.020 [−0.042, 0.003] |

In the negative control the probe's AUROC is 0.497 [0.455, 0.539], and the combined monitor is
statistically indistinguishable from behaviour alone. The behavioural rows are identical across the
two controls because the behaviour is identical; only the synthetic activations differ. This is the
result the pipeline must produce if the evaluation is free of leakage.

The layer sweep in the positive control recovers the planted structure. Validation AUROC is near
chance at layers 0–1 and rises to 0.81 at layers 4–6, consistent with the simulator's signal peaking
at layer 5.

### 5.2 Early warning (RQ3)

The table gives AUROC of states *k* steps before onset against all states from clean trajectories
(horizon-3 monitors):

| k | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Behavioural | 0.777 | 0.713 | 0.595 | 0.507 | 0.432 |
| Combined | 0.869 | 0.783 | 0.695 | 0.623 | 0.550 |

Separability falls monotonically with distance to failure, as H3 predicts. The behavioural monitor
reaches chance three steps out, while the combined monitor stays above chance about one step longer.

At the validation threshold, the combined monitor:

- flags 86% of failing test trajectories at or before onset, and 64% at least one step early (median
  lead 1 step);
- raises at least one false alarm on 26% of clean trajectories.

The behavioural monitor raises false alarms on 51% of clean trajectories.

### 5.3 Generalisation (RQ5)

Leave-one-environment-out AUROC (horizon 3):

| Held out | Behavioural | Probe | Combined |
|---|---|---|---|
| Goal drift | 0.687 | 0.801 | 0.803 |
| Prompt injection | 0.657 | 0.776 | 0.764 |
| Tool safety | 0.658 | 0.751 | 0.761 |

The behavioural monitor degrades by about 0.02–0.05 relative to the in-distribution test (0.704). The
lexical cue features carry over imperfectly between categories. Activation-based monitors transfer
well, which is expected here because the simulator's risk direction is shared across categories. Real
models need not have that property.

### 5.4 Interventions (RQ4)

140 paired runs per condition (70 test instances × 2 seeds). The monitor threshold is 0.590:

| Condition | Violation rate [95% CI] | Task success | Runs intervened | Unsafe / legitimate actions stopped |
|---|---|---|---|---|
| None | 0.457 [0.414, 0.507] | 0.986 | 0.00 | – |
| Warning (monitor) | 0.157 [0.114, 0.200] | 1.000 | 0.54 | – |
| Gate (monitor) | 0.186 [0.100, 0.300] | 0.814 | 0.54 | 33 / 24 |
| Reconsider (monitor) | 0.136 [0.079, 0.179] | 0.993 | 0.54 | 20 / 69 |
| Warning (random, rate-matched) | 0.343 | 0.986 | 0.57 | – |
| Gate (random, rate-matched) | 0.379 | 0.793 | 0.65 | 9 / 27 |
| Reconsider (random, rate-matched) | 0.329 | 0.979 | 0.53 | 3 / 96 |
| Gate (always) | 0.071 | **0.000** | 1.00 | 40 / 138 |

Interpretation:

- **Monitored vs. random.** Monitor-triggered interventions remove 27–32 pp of violations, against
  8–13 pp for random triggers firing at the same rate. Timing is what matters, not frequency alone.
- **Gate cost.** The approval gate costs 17 pp of task success, because 24 deferred actions were
  legitimate.
- **Refuse-all reference.** `gate:always` shows the degenerate end of the trade-off: the fewest
  violations and zero task success.
- **Warning and reconsideration.** Neither costs task success. This rests on the simulator's assumption
  that a warned agent resumes its plan. Measuring that cost is exactly what a real-model run would
  answer.

## 6. Discussion

The framework behaves as a measurement instrument should:

- it detects an effect that exists (positive control);
- it reports none where none exists (negative control);
- it separates useful from merely frequent interventions (rate-matched baselines);
- it exposes trade-offs that a single safety number hides (the gate's task-success cost).

Two analysis choices proved important:

1. **Family-level splits.** Splitting by row would let templates seen in training inflate test scores.
2. **Per-intervention operating points.** The validation max-F1 threshold suits a cheap warning, but a
   gate needs a stricter operating point to avoid blocking legitimate work.

## 7. Limitations

See `docs/limitations.md`. The most important points:

- **No real-model results.** The simulator's effect sizes are planted, so the numbers above are
  evidence about the instrument, not about LLMs.
- **Too few test clusters.** Only 5 test templates means the template-level bootstrap resamples 5
  clusters, and its CIs are likely too narrow. Grouped cross-fitting over all families is the planned
  fix.
- **Exposure confound.** Hazards are short and lexically marked, which favours cue-based behavioural
  features, and seeing the hazard is itself predictive. Analysis restricted to post-exposure states is
  planned.
- **Probes are not causal.** Probe results are correlational and do not identify mechanisms.

## 8. Conclusion and next steps

AFO provides a complete, tested, reproducible pipeline for prospective failure prediction and
monitor-triggered intervention in LLM agents, validated with positive and negative controls. Next
steps:

1. Run `qwen_small.yaml`, then a 3–8B instruction model, on a GPU and report real RQ1–RQ5 results,
   whatever their sign.
2. Add grouped cross-fitting and a post-exposure-only analysis.
3. Generate and validate more templates, including subtler injections.
4. Run activation-steering interventions to test causal claims.
