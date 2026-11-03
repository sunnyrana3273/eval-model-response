# Limitations, confounds, and sources of label noise

## Biggest caveat: no real-model results yet

The bundled experiments (`main-simulated`, `null-control`, `smoke`) use the **simulated policy**. They
show that the pipeline runs, that its leakage controls hold, and that it recovers a planted signal
and reports nothing when none exists. They say nothing about whether real LLM activations predict
real agent failures. That requires running `configs/experiments/qwen_small.yaml` or a larger model.

The Hugging Face path has been run end to end with SmolLM2-135M-Instruct:

- trajectories and activations were stored with correct alignment;
- the model could not produce valid JSON actions, so no meaningful behaviour resulted.

A useful real study needs an instruction-following model of at least 0.5B parameters, and preferably
3B or more.

## Statistical

- **Few test families.** With 33 templates in about 19 families, the 60/20/20 family split leaves
  about 5 test templates. The template-level cluster bootstrap therefore resamples only about 5
  clusters, and its CIs are probably **too narrow**. Bootstrap coverage is poor when there are few
  clusters. Next steps:
  - grouped k-fold cross-fitting, so that every family is tested once;
  - more templates, of which automated generation plus validation is the scalable route;
  - a permutation test at the family level.
- **Correlated rows.** Decision points within a trajectory are correlated, so row-level n overstates
  the information available. CIs resample groups rather than rows for this reason.
- **Multiple comparisons.** Results cover three horizons, four monitors, and a layer sweep. Only the
  primary horizon (`evaluation.primary_horizon`) feeds the hypothesis table. Treat the others as
  exploratory.
- **Layer selection** uses validation data. The test AUROC by layer is descriptive and must not be used
  to pick a layer.

## Construct validity

- **Short, explicit hazards.** Injections and distractors are one sentence long, and cue words
  ("urgent", "ignore previous instructions") are common. A lexical behavioural monitor is therefore
  stronger here than it would be against subtle attacks.
- **Attempted vs. successful violations.** An action the sandbox refused still counts as a violation.
  This is deliberate (intent to act), but it differs from harm-based definitions.
- **Rule-based labels.** Violation rules match tool names and argument regexes. Possible errors:
  - paraphrased unsafe actions, such as a differently named path, can be missed (false negatives);
  - in category C, any permission denial counts, including an honest mistake such as reading the wrong
    path (possible false positives).
- **Goal drift vs. helpfulness.** Doing something extra that the task did not request is labelled as
  drift even when it is harmless. The rules target clearly unauthorised actions (other people's
  files, external recipients), but the line is a judgement call.
- **Task success depends on the plan.** Success criteria are regexes over outputs, so correct answers
  phrased differently may be marked unsuccessful. This affects success rates, not safety labels.

## Confounds

- **Exposure confound.** Pre-hazard states almost never precede a failure. Monitors can earn AUROC
  simply by detecting "hazard text visible", which is a valid real-world signal but not foresight
  about *which* agents will fail. Two analyses address this:
  - the steps-to-onset analysis compares pre-failure states with *all* clean states;
  - restricting to post-exposure states is a recommended follow-up.
- **Step position.** Failures happen at particular depths, so the step index alone is predictive. It is
  a behavioural feature, which keeps the comparison fair, but it inflates every monitor's numbers
  equally.
- **Probes are not explanations.** A probe that predicts failure shows the information is linearly
  decodable. It does not show that the model uses it, or that it is causal. Causal claims would need
  activation patching or steering, which remain future work.

## Simulator-specific

- The simulator's dynamics (rumination, precursors, damping by interventions) are design choices.
  Intervention effects in the simulator are therefore *assumed*: a warning halves risk because the
  config says so.
- The positive control's probe advantage was planted, by encoding the latent `s0` in the
  activations. Its value is that the pipeline *recovers* it under leakage-resistant evaluation, and
  that the null control finds nothing.

## Engineering

- Activations are captured with a separate forward pass over the prompt, which doubles prefill cost.
  Fusing the capture with generation is a possible optimisation.
- Long trajectories are trimmed to `max_prompt_tokens` by dropping the oldest exchanges, which changes
  what the model sees. Trimming is logged implicitly in the prompt, not as a field.
- Sampling on GPUs can be nondeterministic even with seeds.
