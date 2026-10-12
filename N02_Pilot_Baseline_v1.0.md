# N02 Pilot Baseline v1.0

**State:** exploratory baseline frozen for traceability; N02 scientific hypothesis status remains `INCONCLUSIVE`.

## Scope and provenance

- Models: `gpt2` and `EleutherAI/pythia-160m`, 12 Transformer blocks each in the reported run.
- Pilot: 5 prompts per model (general prose, technical prose, academic prose, question/dialogue, code+natural language).
- The pilot is for measurement debugging and hypothesis formation only. These five prompts are excluded from Main Evaluation.
- Layer-wise values below are the supplied five-prompt summary; `n_prompts=5` at every block. They are descriptive only, not inferential estimates.
- The current extraction hooks capture the first-block input and each block output before final LayerNorm. Final LayerNorm is logged separately.

## Core observables frozen for N02

1. **Length-adjusted update norm:** `||h_(l+1) - h_l||_F / sqrt(T)`; store raw Frobenius norm as a diagnostic.
2. **Input–update cosine:** `cos(h_l, Δh_l)`. For scalar-only stored norms it can be recovered from `(||h_(l+1)||² - ||h_l||² - ||Δh_l||²) / (2 ||h_l|| ||Δh_l||)`; the next implementation should prefer the tensor inner product when tensor access is available.
3. **Output/input norm ratio:** `||h_(l+1)||_F / ||h_l||_F`; token count cancels for a given text.

Block outputs are measured before final LayerNorm; final-LN output is a separate diagnostic. Raw hidden-state scales are not literal energy.

## Layer-wise pilot summary

| Model | Block | Mean cosine | Median cosine | Mean update/input norm | Mean output/input norm | n texts |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 0 | 0.0923 | 0.1020 | 11.1945 | 11.3307 | 5 |
| GPT-2 | 1 | 0.4098 | 0.4348 | 2.0234 | 2.6171 | 5 |
| GPT-2 | 2 | 0.2358 | 0.2536 | 3.3507 | 3.7178 | 5 |
| GPT-2 | 3 | 0.8696 | 0.9253 | 0.0797 | 1.0683 | 5 |
| GPT-2 | 4 | 0.8510 | 0.9142 | 0.0646 | 1.0535 | 5 |
| GPT-2 | 5 | 0.7161 | 0.8012 | 0.0516 | 1.0332 | 5 |
| GPT-2 | 6 | 0.5338 | 0.5829 | 0.0446 | 1.0205 | 5 |
| GPT-2 | 7 | 0.3264 | 0.3514 | 0.0500 | 1.0144 | 5 |
| GPT-2 | 8 | 0.2095 | 0.2222 | 0.0572 | 1.0125 | 5 |
| GPT-2 | 9 | 0.1410 | 0.1464 | 0.0761 | 1.0150 | 5 |
| GPT-2 | 10 | 0.1354 | 0.1036 | 0.1552 | 1.0390 | 5 |
| GPT-2 | 11 | -0.7686 | -0.8765 | 0.9656 | 0.6309 | 5 |
| Pythia-160M | 0 | 0.0729 | 0.0695 | 15.1212 | 15.2271 | 5 |
| Pythia-160M | 1 | 0.1981 | 0.2057 | 1.2162 | 1.7361 | 5 |
| Pythia-160M | 2 | 0.3768 | 0.5583 | 1.0953 | 1.7529 | 5 |
| Pythia-160M | 3 | 0.5628 | 0.6425 | 2.3851 | 3.0631 | 5 |
| Pythia-160M | 4 | 0.2774 | 0.3213 | 0.0963 | 1.0269 | 5 |
| Pythia-160M | 5 | 0.1245 | 0.1449 | 0.1044 | 1.0156 | 5 |
| Pythia-160M | 6 | 0.0220 | 0.0389 | 0.1056 | 1.0074 | 5 |
| Pythia-160M | 7 | 0.0779 | 0.0899 | 0.1383 | 1.0210 | 5 |
| Pythia-160M | 8 | -0.4252 | -0.4166 | 0.1631 | 0.9529 | 5 |
| Pythia-160M | 9 | -0.7207 | -0.7838 | 0.2600 | 0.8486 | 5 |
| Pythia-160M | 10 | -0.8402 | -0.8853 | 0.4998 | 0.6456 | 5 |
| Pythia-160M | 11 | -0.3785 | -0.4543 | 2.1735 | 2.0271 | 5 |

## Final-block branch decomposition (five prompts)

| Model | Prompt | Tokens | Input norm | Attention norm | MLP norm | Branch cosine | Update norm | Output/input ratio | Max reconstruction error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | P1_general | 9 | 3196.979 | 2717.872 | 357.245 | 0.366 | 2868.069 | 0.419 | 1.50e-05 |
| gpt2 | P2_technical | 9 | 3200.423 | 2726.625 | 347.067 | 0.366 | 2871.947 | 0.422 | 1.20e-05 |
| gpt2 | P3_academic | 15 | 3214.751 | 2738.920 | 379.905 | 0.392 | 2908.925 | 0.485 | 1.40e-05 |
| gpt2 | P4_question | 13 | 3192.753 | 2731.693 | 404.561 | 0.374 | 2907.237 | 0.483 | 1.30e-05 |
| gpt2 | P5_code_mixed | 127 | 3876.863 | 3967.207 | 1371.641 | 0.431 | 4722.941 | 1.345 | 2.30e-05 |
| pythia-160m | P1_general | 9 | 220.665 | 58.621 | 330.531 | 0.723 | 375.096 | 1.465 | 3.00e-06 |
| pythia-160m | P2_technical | 9 | 219.340 | 57.949 | 348.995 | 0.757 | 394.683 | 1.571 | 2.00e-06 |
| pythia-160m | P3_academic | 15 | 231.168 | 77.502 | 374.363 | 0.759 | 436.088 | 1.687 | 2.00e-06 |
| pythia-160m | P4_question | 11 | 143.711 | 61.916 | 343.572 | 0.876 | 398.913 | 2.737 | 3.00e-06 |
| pythia-160m | P5_code_mixed | 92 | 342.792 | 186.496 | 759.779 | 0.877 | 927.603 | 2.676 | 5.00e-06 |

## Exploratory interpretation (not a claim of generality)

- GPT-2 final block mean input–update cosine was −0.7686 across five prompts; mean output/input norm ratio was 0.6309. General prose/question/academic prompts showed a ratio below 0.5; the code-mixed prompt behaved differently (ratio 1.3451).
- Pythia-160M mean cosine became negative in blocks 8–11 (minimum −0.8402 in block 10) and was −0.3785 in block 11; output/input norm ratio at block 11 averaged 2.0271.
- For the general prose prompt, GPT-2 final-block Attention branch norm (2717.9) was much larger than MLP branch norm (357.2), while for Pythia the MLP branch norm (330.5) exceeded Attention (58.6).
- Maximum reconstruction errors for the final-block branch decomposition in the supplied pilot were approximately 1.2e−5–2.3e−5 for GPT-2 and 2e−6–5e−6 for Pythia. This supports numerical consistency for these specific runs, not architecture-wide validation.
- The code-mixed texts are substantially longer than prose prompts (127 tokens for GPT-2 and 92 for Pythia in this run). Length and content are confounded with style; do not infer a code-style effect from this pilot.

## Frozen decision

- **Pilot baseline:** recorded and frozen for traceability.
- **N02 scientific hypothesis:** `INCONCLUSIVE`.
- **Main Evaluation:** use the separate candidate manifest and preflight notebook below. Do not include pilot prompts.
- **Resampling unit:** the main manifest has four style renditions per semantic topic. For within-topic style contrasts, resample `topic_id` clusters (100 clusters), not the 400 rendition rows independently.
- **Before activation capture:** review text quality, run tokenizer length audit for both model tokenizers, revise candidates using text/tokenizer information only, freeze the final manifest and its SHA-256 before collecting activations.
