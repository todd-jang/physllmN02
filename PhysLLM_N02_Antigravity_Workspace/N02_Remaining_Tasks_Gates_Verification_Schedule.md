# physllmN02 잔여 Task / Gate / 검증 / 스케줄

## 1. 현황 요약
- 데이터셋: frozen manifest v1.0.3 (400 rows) — SHA/텍스트 지문 일치 확인됨
- 활성화 메트릭: `results/activations/N02_activation_metrics.csv` = **44,826행**, **6모델** 모두 정합식 통과
  - gpt2(12L/768,MHA+LN), pythia-160m(12L/768,MHA+LN), qwen2.5-0.5b(24L/896,GQA+RMS), qwen2.5-1.5b(28L/1536,GQA+RMS), qwen2.5-3b(36L/2048,GQA+RMS), gemma-2-2b(26L/2304,GQA+RMS, soft-cap/sliding) — gemma는 T001 단일텍스트 26행(제한적)
- 분석: `reports/analysis/N02_cross_model_*.csv/json` 생성 완료(cross_model_analysis.py)
- 범위문서: `N02_Architecture_Scope_v1.1.md` 갱신(6모델 측정 상태 반영)
- 커밋: c2cf903(최신)

## 2. 잔여 Task

### T1. gemma-2-2b 전체 400텍스트 캡처(재실행)
- 목적: gemma를 "제한적(26행)" → "400×26" 완전본으로 올리기
- 원인: (5).zip 내 gemma 분기에서 컬럼/계산 이슈로 26행만 유효 → 유효분만 추출
- 방법: Colab L4 nb04의 gemma 분기 재검증
  - layer 접근: `transformer.layers`/`model.layers` 확인
  - 텐서쌍: `h_l`(입력), `h_{l+1}`(출력), `update = h_{l+1}-h_l` 정의 정확화(pre-residual vs residual)
  - RMSNorm 적용 위치, `hidden_states` vs `outputs.last_hidden_state`
  - dtype=float32 고정, grad 없음
- 산출: 완전 gemma CSV(10400행) → 기존 gemma 26행 대체 or 병합 검증 후 교체
- 우선순위: 중간(스케일 축은 5모델로도 충분하나 완전성 위해 권장)

### T2. `update_rms_per_element` 계산/직렬화 재점검(선택 개선)
- 현상: (3)/(5).zip에서 5모델의 `update_rms_per_element`이 빈값→nan으로 직렬화된 사례 존재. 현재는 (3).zip 값 복원+recompute로 정합 통과
- 조치: nb04 write 시 항상 유한값으로 기록(빈문자열 금지) + write 직전 `math.isfinite` 체크

### T3. cross_model_analysis 출력의 rms nan 제거/일관화
- 요약: `global_mean_update_rms`가 일부 케이스에 nan? 현재 summary에 rms 값은 출력되나, 스크립트 내에서 nan 허용 여부 확인
- 범위 경미

## 3. Gate(합격 기준)

### G1. 무결성(필수)
- 매니페스트: `manifest_sha256==841aca61...`, `text_fingerprint==3b3c049b...`, 행수 400
- 메트릭 행수: 각 모델 `400×n_layers` 정확 일치
- 정합식(엄격 2e-2 + 1e-9): 
  - `|upn - updn/sqrt(nt)| <= |upn|*0.02+1e-9`
  - `|rms - updn/sqrt(nt*D)| <= |rms|*0.02+1e-9`
  - `|ratio - outn/max(inn,1e-12)| <= |ratio|*0.02+1e-9`
  - `|cos| <= 1.0+1e-9` and finite
  - `ratio >= -1e-9`

### G2. provenance(필수)
- `results/activations/N02_capture_run_metadata_colab_l4_*.json`에 device/gpu_name/dtype/python/torch/transformers/manifest_sha256/text_fingerprint/models 포함
- `manifest_used.csv` 저장

### G3. 분석 재현성(권장)
- `cross_model_analysis.py` 재실행 시 에러 0, 출력 파일 6종 생성

## 4. 검증 절차(체크리스트)

1. `python3 -m py_compile PhysLLM_N02_Antigravity_Workspace/scripts/capture_activations.py PhysLLM_N02_Antigravity_Workspace/scripts/cross_model_analysis.py`
2. 메트릭 검증(전행): 위 G1 스크립트로 통과 확인
3. 공유모델(gpt2/pythia-160m/qwen2.5-0.5b) 값 일관성 대조(선택)
4. nb04 gemma 분기 테스트: dry-run(CPU)으로 구조 확인 후 Colab L4 실행
5. 완전 gemma 수령 시: 행수 10400, 정합식 통과, 기존 대체 후 cross_model_analysis 재실행

## 5. 스케줄(실제)

- **지금(완료)**: 5모델+gemma(26행) 병합·검증·분석·푸시(c2cf903)
- **단기(1–2일)**: nb04 gemma 수정 → Colab L4 재캡처 → 10400행 검증 → 병합·분석·커밋
- **중기**: 필요시 (5).zip rms 이슈 원인(nb04 write) 수정(PR)
- **병행**: paper-diary Day 5(DeepSeek-V3 2412.19437) — 독립 작업

## 6. 참고 산출물 경로
- 메트릭: `PhysLLM_N02_Antigravity_Workspace/results/activations/N02_activation_metrics.csv`
- 분석: `PhysLLM_N02_Antigravity_Workspace/reports/analysis/`
- 범위: `PhysLLM_N02_Antigravity_Workspace/N02_Architecture_Scope_v1.1.md`
- 노트북: `PhysLLM_N02_Antigravity_Workspace/notebooks/04_colab_l4_qwen_gemma_capture.ipynb`
