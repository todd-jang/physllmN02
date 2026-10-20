# N02 gemma-2-2b 캡처 무효화 및 복구 기록 (2026-10-19)

## 1. 결정
`gemma-2-2b`의 26행 캡처는 **`N02_results_qwen_gemma_capture (5).zip`**에서 유래하며, 이는 raw residual stream이 아니라 **텐서별 정규화본**이다. 따라서 해당 행을 무효화하고, 동일 텍스트(T100)의 **raw 캡처인 `(3).zip` = `(4).zip`**으로 교체한다.

- 근거 1: `(5).zip`의 gemma 26행 전부에서 `input_norm_fro == output_norm_fro == 1.0` (±1e-13).
- 근거 2: 동일 텍스트·동일 레이어(T100, L0, 203 tokens, hidden_dim 2304) 대조.

| 항목 | (3) = (4) raw | (5) normalized |
|---|---|---|
| `input_norm_fro` | 1271.3928002765458 | 0.9999999999999599 |
| `output_norm_fro` | 1104.804908524303 | 0.9999999999999615 |
| `update_norm_fro` | 997.4990225632624 | 0.8298265400628104 |
| `update_rms_per_element` | 1.4585563839617433 | 0.001213383442200552 |
| `input_update_cosine` | −0.5483503101737419 | −0.41491327003141915 |

- 근거 3: `results/activations/N02_activation_metrics.csv.bak`의 SHA-256은 `c9cf0306d98028ad…`로 `(3).zip` 및 `(4).zip`의 metrics CSV와 **바이트 동일**하다. 즉 유효 raw 원본은 이미 워크스페이스에 존재한다.

## 2. 무효 아티팩트

| 아티팩트 | gemma 행 | `in_norm==1.0` | 항등식 잔차 median | cos 범위 | 판정 |
|---|---|---|---|---|---|
| `N02_results_qwen_gemma_capture (1).zip` | 10400 | 0/10400 | 4.15e-01 | 0.9294 ~ 10.1420 | VOID |
| `N02_results_qwen_gemma_capture (2).zip` | 10400 | 0/10400 | 4.15e-01 | 0.9294 ~ 10.1420 | VOID |
| `N02_results_qwen_gemma_capture (5).zip` | 26 | 26/26 | 5.11e-15 | -0.4149 ~ -0.1477 | VOID |
| `N02_results_qwen_gemma_capture.zip` | 10400 | 10400/10400 | 4.52e-02 | 0.0004 ~ 0.0030 | VOID |

`(1)`과 `(2)`는 `|cos|<=1`을 위반하므로(최대 10.1420) 물리적으로 성립하지 않는다. `(5)`와 base는 정규화로 인해 `update_rms_per_element`가 raw 대비 약 1,200배 축소되었다.

## 3. 저장소 내 무효 파일

| 경로 | SHA-256 | 처리 |
|---|---|---|
| `results/activations/N02_activation_metrics.csv` | `4e16bda0d15f0c5b…` | void_in_place |
| `results/activations/N02_activation_metrics_6models_correct_20261018.csv` | `4e16bda0d15f0c5b…` | moved_to_void_dir |
| `results/activations/N02_activation_metrics_6models_20261018.csv` | `fd276875aef81324…` | moved_to_void_dir |
| `results/activations/N02_activation_metrics_gemma_fix_20261018.csv` | `2ba598a5c5e04c03…` | moved_to_void_dir |

`N02_activation_metrics.csv`는 `scripts/cross_model_analysis.py`의 기본 경로이므로 이동하지 않고 **제자리 무효 표기**만 하였다(AGENTS.md: "Do not overwrite source CSVs"). 나머지 3개 변형은 `void_20261019/`로 이동하였다.

## 4. 복구본
- 경로: `results/activations/N02_activation_metrics_6models_gemma_rawfix_20261019.csv` (데이터; `results/**`는 .gitignore 정책상 미추적)
- SHA-256: `b79ffc22310b40618b8ca8c0f9d17d1e5d1f86b70eb1626f81293ab28329cb2e`
- 행 수: 44,826 (비-gemma 44,800행은 정본과 완전 동일, gemma 26행만 교체)
- gemma `in_norm==1.0`: **0/26**
- 항등식 잔차: median `6.96e-15`, max `3.73e-14`
- 행 단위 변경 기록: `data/review/N02_gemma_capture_void_20261019.rowchanges.csv` (182건 = 26행 × 7필드)

## 5. 미해결 사항
1. **400텍스트 raw gemma 캡처는 여전히 부재**하다. 유효분은 T100 1건(26행)뿐이다.
2. `reports/analysis/*`는 무효 gemma 행으로 생성되었으므로 6모델 주장 전에 재생성해야 한다.
3. `N02_capture_run_metadata.json`은 `per_model={}`, 실행 시간 0.06초, tokenizer ID·seed 미기록으로 AGENTS.md 요건을 충족하지 못한다.
4. `(5).zip`이 Google Drive `/content/` 사본이라면 `(3)` 또는 `(4)`로 교체해야 한다.

## 6. 재현
```python
import csv, math
rows = list(csv.DictReader(open("results/activations/N02_activation_metrics_6models_gemma_rawfix_20261019.csv")))
g = [r for r in rows if r["model_key"] == "gemma-2-2b"]
assert all(abs(float(r["input_norm_fro"]) - 1.0) > 1e-9 for r in g)
for r in g:
    i, o = float(r["input_norm_fro"]), float(r["output_norm_fro"])
    d, c = float(r["update_norm_fro"]), float(r["input_update_cosine"])
    assert abs(math.sqrt(i*i + 2*c*i*d + d*d) - o) / o < 1e-12
```

## 7. 위치 규약
`results/**`는 저장소 `.gitignore`에서 제외되어 있으므로, **감사 기록(본 문서·JSON·행 단위 변경 로그)은 `data/review/`에 두고 데이터 산출물만 `results/activations/`에 둔다.** 무효 아티팩트는 `results/activations/void_20261019/`에 보관한다.
