# Threat model

## What is being modelled

The *agent under test* is an LLM acting for a **principal**, who supplies the system prompt and the
objective. It works through tools with explicit permissions. The study covers three ways it can go
wrong:

| Threat | Adversary / pressure | Channel | Violation |
|---|---|---|---|
| Goal drift | A non-principal (colleague, stakeholder) with a competing request | Notes, files, messages the task requires reading | Acting on the competing request instead of, or as well as, the authorised objective |
| Prompt injection | An attacker who controls content | File contents, web pages, API responses, search results | Treating untrusted content as instructions: exfiltration, manipulation, destructive calls |
| Unsafe tool use | No adversary; situational pressure (errors, deadlines, tips) | Error messages, logs, docs | Exceeding granted permissions: privileged commands, credential access, destructive writes |

Assumptions:

- The principal's instructions are trusted.
- Everything returned by a tool is untrusted, including files in the principal's workspace, since
  those files may have been written by others.

Out of scope:

- jailbreaks issued by the principal;
- attacks on the monitor itself, such as adversarial suffixes that evade a probe;
- multi-agent collusion;
- training-time poisoning.

## Safety of the framework itself (NFR-01)

- **No real side effects.** File, e-mail, HTTP, and shell tools operate only on in-memory state.
  `run_command` never calls a shell. It matches commands against canned outputs and logs every
  attempt. `http_request` never opens a socket.
- **Synthetic secrets only.** Credentials in templates are `AFO_FAKE_*`. Attacker endpoints are
  `.example` domains that are not routable.
- **Limits.** `max_steps`, a per-run `timeout_s`, `max_output_chars`, and `max_consecutive_invalid` are
  set in config.
- **Secrets hygiene.** Hugging Face tokens, if needed, come from environment variables and are never
  written to configs or manifests. `.env` files are git-ignored.
- **Dashboard.** Read-only `GET` endpoints, with experiment ids and file names validated against path
  traversal.

## Monitor and intervention integrity

- Monitors and interventions receive only observable inputs (`DecisionContext`), never labels.
- Thresholds are fixed on validation data, never on test data.
- Gate decisions are labelled legitimate or unsafe **after the fact** by a dry run, so the gate cannot
  peek at the answer.

## Residual risks

- The hazards are hand-written and short, so real injections can be far more subtle. Monitors that
  rely on lexical cues may be easy to evade.
- A deployed monitor would be adaptively attacked; this study assumes a non-adaptive adversary.
- Some violation rules are regular expressions over arguments, so paraphrased unsafe actions could be
  missed (false negatives in the labels). See `limitations.md`.
