# PhysLLM N02 Workspace Rules

- Treat all files under `data/candidate/` and `data/history/` as immutable source artifacts.
- Never label a manifest frozen unless a separate, human-approved release checklist exists and every required gate is evidenced.
- Never auto-complete `human_review_status` or infer semantic correctness from syntax/length checks.
- Do not overwrite source CSVs. Write repaired candidates to a new, versioned file and log row-level changes.
- Any analysis artifact must record manifest SHA-256, code revision/hash, model ID, tokenizer ID, package versions, seed, and run timestamp.
- Use topic-level clustered inference. Do not treat tokens as independent samples.
- Distinguish exploratory outputs from confirmatory results.
- Review generated diffs and run tests before accepting agent edits.
- For Colab, do not install/upgrade PyTorch blindly; inspect the existing runtime first.
- Avoid loading 20B/120B models in integrity notebooks. Dataset audit must be CPU-only unless tokenizer checks are explicitly requested.
- N02 scope is limited to MHA + Dense FFN architectures (see `N02_Architecture_Scope_v1.0.md`). MoE and MLA models are deferred to N03.
- Never compare raw Frobenius norms across models with different `hidden_dim`. Use dimensionless ratios (output/input norm ratio, cosine) or RMS-per-element normalization.
- For cross-depth comparisons, always use relative depth `rho_block = l/(L-1)`, never raw `layer_idx` across models with different layer counts.
