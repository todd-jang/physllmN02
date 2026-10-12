# N02 Manifest Version & Provenance Note

**Decision:** No historical freeze claim is accepted as the current release approval. The active manifest in this workspace is a candidate only.

| Artifact | Role in this workspace | SHA-256 of exact copied file |
|---|---|---|
| `N02_Main_Evaluation_CANDIDATE_Manifest_v1.0.3.csv` | Current audit target; candidate, not frozen | `4d2a7590b2c13310aa798fd89bb2a7f07c3fa26696967c0fcfc025b236b63158` |
| `N02_Main_Evaluation_Candidate_Manifest_v1.0.csv` | Earlier candidate preserved for comparison | `94ac3def32afc651492f77d36df0ea7bd0aa839f68c2c5e4efe4a62f195a0a80` |
| `N02_Frozen_Manifest_Audit_v1.0.1.zip` | Historical artifact; contains a report stating the manifest was frozen | `c673e4beb1113a11fed21575647aaca197635eb7920175761da6ab83041b9651` |
| `N02_Frozen_Manifest_Audit_v1_0_3.zip` | Historical v1.0.3 candidate + preflight report | `0b2f4b52ebcdeac1066c0b2a52111770fd8bc62b67ddd92c280b23de3e254e72` |

## Why the old freeze claim is not carried forward

- The v1.0.1 preflight report says `human_content_review_complete: true` and `frozen_for_confirmatory_analysis: true`.
- A separate v1.0.3 report says `human_review_of_replaced_items_done: false` and `frozen_for_confirmatory_analysis: false`.
- The current v1.0.3 source file also differs from the earlier v1.0 candidate at the byte level and includes a different set of precomputed token-count fields.
- The v1.0.3 preflight JSON inside the historical ZIP records `source_manifest_file_sha256: a9f4a23c631857dcbe80c76e3ee6732ee956ce5b007b5e1ea542da7b09ca8f48`, while the exact v1.0.3 CSV in that ZIP has SHA-256 `4d2a7590b2c13310aa798fd89bb2a7f07c3fa26696967c0fcfc025b236b63158`. The relationship between that source hash and the enriched candidate file is not established by the surviving artifacts.
- The currently preserved artifacts do not constitute a complete, row-level reviewer ledger that can independently substantiate the older content-review-complete claim.

Therefore, previous freeze fields are recorded as historical claims, not relied upon as present-day approval. Both candidate CSVs and both historical ZIPs are kept unchanged. The new independent audit produces its own report and never mutates the source data.

## Current machine audit baseline

The script's initial structural audit of the current v1.0.3 candidate found no hard structural failure among the checks it performs. It found 31 normalized Python-code templates among 100 code-format rows; 19 normalized templates occur at least three times and the largest group occurs seven times. These are review-priority signals, not automatic proof of invalidity. All 400 rows remain marked for semantic review. Independent tokenizer retokenization has not yet been run in this baseline.
