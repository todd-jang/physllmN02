# N02 Architecture Scope Declaration v1.1 (검토·교정본)

> **v1.0(`N02_Architecture_Scope_v1.0.md`)을 검토·교정한 버전입니다. 가설 번호 라벨(H1/H2/...)은 사용하지 않습니다.**
> 확정일: 2026-10-18 · 상태: `REVIEWED_DRAFT` (인간 승인 전)

## 1. 범위 (In-Scope)

N02 확증 분석은 **dense attention(MHA/GQA) + dense feed-forward FFN** Transformer로 한정한다.

### 1.1 측정 완료 모델 (확증 분석에 사용 가능)

| Model Key | HF ID | Attention | Norm | Residual | Layers | Hidden | 측정 |
|---|---|---|---|---|---|---|---|
| `gpt2` | `gpt2` | MHA | LayerNorm | serial (pre-LN) | 12 | 768 | ✅ 400×12 |
| `pythia-160m` | `EleutherAI/pythia-160m` | MHA | LayerNorm | parallel | 12 | 768 | ✅ 400×12 |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` | GQA | RMSNorm | serial (pre-RMSNorm) | 24 | 896 | ✅ 400×24 |

> **2026-10-18 갱신 — 스케일 축 캡처 준비.** `qwen2.5-1.5b`, `qwen2.5-3b`, `gemma-2-2b`는 로컬(M1 8GB)에서 fp32 실행이 불가하므로 **Colab L4 런타임 전용 노트북** `notebooks/04_colab_l4_qwen_gemma_capture.ipynb`(자체 포함 · 무결성 게이트 내장)로 캡처한다. 이 4종은 **아직 측정 완료가 아니며**, 캡처 + 검증 + provenance 확정 이후에만 사용 가능하다.

- 캡처 총량: **19,200 관측** (400 text × {12, 12, 24} layers).

## 2. 범위 내 후보 (측정 전 — 주장 금지)

아래 모델은 범위에는 들지만 **아직 측정되지 않았습니다.** 측정이 완료되어 동일 audit/해시 절차를 통과하기 전에는 **어떤 분석·주장에도 사용하지 않습니다.**

- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-3B`
- `google/gemma-2-2b`  ← Gemma2는 sliding-window local/global 교대 attention + logit soft-cap을 사용하므로, 사용 시 "dense attention" 가정의 예외로 별도 표기

## 3. 범위 밖 (Out-of-Scope → N03으로 분리)

| 아키텍처 | 예 | N03 이관 사유 |
|---|---|---|
| MoE (shared/routed experts) | DeepSeek-MoE, Mixtral, GPT-OSS | routing-conditional branch 분해 필요; sparse MLP는 token별 이산 변동 → 텍스트당 평균이 라우팅 분산을 소거 |
| MLA (latent attention) | DeepSeek-V2/V3 | latent 관측량(`c^KV` 노름 등) 추출 로직 신규 필요 |
| SSM / 하이브리드 | Mamba, Jamba | 잔차 스트림 동역학 프레임워크 적용 불가 |

## 4. 교차 아키텍처 비교 규칙 (필수)

> 서로 다른 `hidden_dim` 또는 `n_layers` 모델 간 **원시 Frobenius 노름 절대값 비교는 금지**한다.

**허용**
- 상대 깊이 `rho_block = l / (L-1)` (raw `layer_idx` 교차 비교 금지)
- 무차원 비율: `input_update_cosine`, `output_input_norm_ratio`
- `update_rms_per_element` (`sqrt(D)` 정규화)

**금지**
- `hidden_dim`이 다른 모델 간 raw 노름 절대값 비교
- 결합 모형에서 노름을 직접 비교 (아키텍처는 교란 요인으로 취급)

## 5. 측정 상태 매트릭스 (정직한 커버리지)

| 계열 | 모델 | 측정 | 사용 가능 범위 |
|---|---|---|---|
| MHA + LayerNorm + serial | gpt2 | ✅ 400×12 | 확증 분석 |
| MHA + LayerNorm + parallel | pythia-160m | ✅ 400×12 | 확증 분석 |
| GQA + RMSNorm + serial | qwen2.5-0.5b | ✅ 400×24 | 확증 분석 |
| GQA + RMSNorm | qwen2.5-1.5b / 3B | ✅ 400×28 / 400×36 (Colab L4 fp32) | 스케일 축 확증 (5모델 병합) |
| GQA + RMSNorm (soft-cap/sliding) | gemma-2-2b | ⚠️ 캡처됨(Colab L4) — 행정합식 이상 발견 → 재캡처 필요 | gemma 분기 별도 검증 전 미사용 |

## 6. v1.0 대비 교정 사항

1. **커버리지 과대표기 제거**: v1.0 §1 표가 6모델을 "in-scope"로 나열해 측정된 것처럼 보이게 함 → 본 §2에서 **미측정 모델을 명시적으로 분리**.
2. **가설 커버리지 매트릭스 제거**: v1.0 §4가 미측정 모델(Qwen 1.5b/3B, Gemma)을 근거로 검증 "가능"을 표기 → **데이터 없는 주장**이므로 §5 측정 상태로 대체.
2. **표현 정합**: v1.0 §1 제목 "MHA + Dense FFN" vs 내용(GQA/RMSNorm 포함) 불일치 → 제목을 "dense attention(MHA/GQA) + dense FFN"으로 정정.
3. **결론 스코프 문구**: 모든 결론은 "**측정된 dense-attention 모델군(gpt2, pythia-160m, qwen2.5-0.5b)에서**"로 명시한다.

## 7. 핸드오프

MoE routing 분해, MLA latent 관측량, SSM 대응은 **N03** 설계 문서에서 다룬다.
