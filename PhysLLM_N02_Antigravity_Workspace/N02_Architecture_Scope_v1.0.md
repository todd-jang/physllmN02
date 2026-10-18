# N02 Architecture Scope Declaration v1.0

**Effective from:** 2026-10-18 (post-freeze Day 1)
**Applies to:** All N02 confirmatory analyses using `N02_Main_Evaluation_FROZEN_Manifest_v1.0.3.csv`

## 1. In-Scope Architectures (MHA + Dense FFN)

N02 confirmatory experiments are limited to **Multi-Head Attention (MHA/GQA) + Dense Feed-Forward** transformer architectures. The following models are registered:

| Model Key | HuggingFace ID | Architecture | Norm Type | Attention | Residual | Layers | Hidden Dim |
|-----------|---------------|-------------|-----------|-----------|----------|--------|------------|
| `gpt2` | `gpt2` | GPT-2 | LayerNorm | MHA | Serial (pre-LN) | 12 | 768 |
| `pythia-160m` | `EleutherAI/pythia-160m` | GPT-NeoX | LayerNorm | MHA | Parallel | 12 | 768 |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` | Qwen2 | RMSNorm | GQA | Serial (pre-RMS) | 24 | 896 |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B` | Qwen2 | RMSNorm | GQA | Serial (pre-RMS) | 28 | 1536 |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B` | Qwen2 | RMSNorm | GQA | Serial (pre-RMS) | 36 | 2048 |
| `gemma-2-2b` | `google/gemma-2-2b` | Gemma2 | RMSNorm | GQA | Serial (pre-RMS) | 26 | 2304 |

## 2. Out-of-Scope for N02

The following architectures are explicitly **excluded** from N02 and deferred to N03:

- **MoE (Mixture of Experts):** GPT-OSS 120B, Mixtral, DeepSeek-MoE 등
  - Reason: routing-conditional branch decomposition 요구, capture 파이프라인 확장 필요
- **MLA (Multi-head Latent Attention):** DeepSeek-V2 등
  - Reason: latent 관측량 추출 로직 신규 개발 필요
- **SSM / Mamba 아키텍처:** Mamba, Jamba 등
  - Reason: 트랜스포머 잔차 스트림 동역학 프레임워크 적용 불가

## 3. Cross-Architecture Comparison Rules (필수)

> **핵심 원칙:** 서로 다른 `hidden_dim` 또는 `n_layers`를 가진 모델 간 **원시 Frobenius 노름의 절대값 비교는 금지**됩니다.

### 3.1 허용되는 비교 방식

| 비교 유형 | 정규화 방법 | 예시 |
|----------|-----------|------|
| 레이어 위치별 프로파일 | **상대 깊이** `rho_block = l / (L-1)` 사용 | 12-layer vs 24-layer 비교 시 rho 기준 보간 |
| 놈 변동 | **Output/Input norm ratio** (무차원 비) | `||h_{l+1}||_F / ||h_l||_F` — dim 독립 |
| 놈 크기 | **RMS per element** `||Δh||_F / sqrt(T × D)` | hidden_dim 차이 정규화 |
| 방향 변화 | **Input-update cosine** (무차원) | `cos(h_l, Δh_l)` — 항상 비교 가능 |
| 스타일 효과 | **클러스터 내 분산비** (F-ratio) | 절대 놈이 아닌 분산 구조 비교 |

### 3.2 금지되는 비교 방식

- ❌ `input_norm_fro(gpt2)` vs `input_norm_fro(qwen2.5-0.5b)` 직접 비교
- ❌ `update_norm_per_token`의 모델 간 절대값 순위 비교 (dim 미보정)
- ❌ 서로 다른 `n_layers` 모델의 `layer_idx` 직접 대응 (rho 미사용)

### 3.3 같은 아키텍처 계열 내 스케일 비교 (H2 검증)

동일 아키텍처 계열(예: Qwen2.5 {0.5B, 1.5B, 3B}) 내에서는:
- `update_rms_per_element` 기준 스케일 트렌드 분석 가능
- `rho_block` 기준 프로파일 형태(shape) 보존성 검증 가능
- 절대 놈 비교는 여전히 `sqrt(D)` 정규화 필요

## 4. Hypothesis Coverage Matrix

| 가설 | N02 검증 가능 여부 | 근거 |
|------|-------------------|------|
| H1: MoE 라우팅 안정성 | ❌ N03 이관 | MoE 모델 미포함 |
| H2: 파라미터 스케일 불변량 | ✅ 가능 | Qwen2.5 {0.5B, 1.5B, 3B} 3점 비교 |
| H3: RMSNorm 엔트로피 경계 | ✅ 가능 | LayerNorm (GPT-2, Pythia) vs RMSNorm (Qwen, Gemma) |
| H4: 토큰 타입 등방성 보존 | ✅ 가능 | 4스타일 × 6모델 교차 검증 |

## 5. Decision Record

- **Decision:** N02 scope = MHA + Dense FFN only (Option B confirmed)
- **Date:** 2026-10-18
- **Rationale:** 동결 완료 후 매니페스트 변경 없이 진행 가능; MoE 120B급은 현재 인프라에서 capture 불가; MHA+dense 범위 내에서 H2/H3/H4 완전 검증 가능
- **Follow-up:** N03 로드맵에 MoE routing-conditional decomposition 및 MLA latent observable 확장 포함
