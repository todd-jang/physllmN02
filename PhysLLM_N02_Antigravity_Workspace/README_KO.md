# PhysLLM N02 — Antigravity IDE + Google Colab Workspace

**현재 상태: CANDIDATE / NOT FROZEN.** 이 패키지는 실험 편집과 감사 절차를 준비하지만 데이터셋을 동결하거나 인간 검토 완료를 선언하지 않습니다.

## 목표

- Google Antigravity IDE에서 프로젝트 코드·CSV·노트북을 한 workspace로 관리
- Google Colab 원격 런타임을 노트북 커널로 사용
- N02 후보 입력문 400개를 구조·중복·길이·코드 문법 관점에서 재현 가능하게 감사
- 내용 검토와 동결 승인을 자동 검사와 분리
- 확인 실험 결과의 출처를 manifest SHA-256으로 추적

## 1. Antigravity IDE에서 열기

1. ZIP을 로컬 폴더에 압축 해제합니다.
2. Antigravity IDE에서 `Open Folder`로 `PhysLLM_N02_Antigravity_Workspace` 폴더를 엽니다.
3. 확장(Extensions)에서 다음을 설치하거나 사용 가능 여부를 확인합니다.
   - Google Colab (`Google.colab`)
   - Jupyter (`ms-toolsai.jupyter`)
   - Python (`ms-python.python`)
4. `notebooks/00_colab_runtime_check.ipynb`를 엽니다.
5. `Select Kernel` → `Colab` → `Auto Connect`를 선택하고 계정 승인을 진행합니다.

> 호환성 주의: Colab 확장의 공식 안내는 VS Code를 대상으로 합니다. Antigravity IDE에서 해당 확장이 설치·실행되는지는 IDE/마켓플레이스 환경에 따라 달라질 수 있습니다. 선택 항목이 보이지 않으면 같은 `.ipynb` 파일을 Google Colab 웹에서 열어 실행하세요. 이 workspace는 특정 확장 설치 성공을 보장하지 않습니다.

공식 시작 안내: https://github.com/googlecolab/colab-vscode

## 2. Colab 원격 런타임과 파일을 연결하기

원격 Colab 커널이 로컬 파일을 자동으로 읽는다고 가정하지 않습니다. **추천 방식은 Google Drive for desktop 동기화**입니다.

1. ZIP을 로컬에서 압축 해제합니다.
2. 폴더 `PhysLLM_N02_Antigravity_Workspace`를 Google Drive의 `My Drive` 안으로 옮기거나 복사합니다.
3. Google Drive for desktop을 사용하는 경우, 동기화된 `My Drive/PhysLLM_N02_Antigravity_Workspace` 폴더를 Antigravity IDE에서 직접 엽니다. 로컬에서 수정한 파일이 Drive와 동기화되면 Colab 커널이 최신 내용을 읽을 수 있습니다.
4. Colab에서 Drive를 연결하고 `notebooks/01_manifest_integrity_audit.ipynb`를 실행합니다.
5. 노트북 상단 `PROJECT_ROOT`가 다음을 가리키는지 확인합니다.
   `/content/drive/MyDrive/PhysLLM_N02_Antigravity_Workspace`

Google Drive for desktop을 사용하지 않는 경우에도 실험할 수 있지만, 로컬 IDE에서 `scripts/audit_manifest.py`를 수정한 뒤 실행하기 전에 업데이트한 파일/폴더를 Drive에 다시 업로드해야 합니다. 동기화 여부는 파일의 수정 시각과 내용을 확인해 주세요.

노트북은 출력을 `reports/audits/` 및 `results/` 아래로 저장합니다. 원본 후보 CSV는 덮어쓰지 않습니다.

## 3. 권장 실행 순서

1. `notebooks/00_colab_runtime_check.ipynb` — Colab 커널/라이브러리/장치 확인. GPU를 쓰지 않는 환경 검사입니다.
2. `notebooks/01_manifest_integrity_audit.ipynb` — 후보 manifest의 독립 감사. 이 단계는 토크나이저 재측정을 선택적으로 수행합니다.
3. `notebooks/02_review_queue_overview.ipynb` — 수동 검토 대기열을 확인합니다.
4. 감사 및 내용 검토가 승인되기 전에는 archive의 활성화 캡처 노트북을 확인 실험으로 실행하지 않습니다.

## 4. 로컬에서 빠른 검증

Python 3.10+ 권장.

```bash
python -m pip install -r requirements-local.txt
python scripts/audit_manifest.py \
  --manifest data/candidate/N02_Main_Evaluation_CANDIDATE_Manifest_v1.0.3.csv \
  --out-dir reports/audits/local_run
python -m pytest -q
```

별도 토크나이저 재측정은 다음과 같이 수행할 수 있습니다(인터넷 연결 및 `transformers` 필요).

```bash
python scripts/audit_manifest.py \
  --manifest data/candidate/N02_Main_Evaluation_CANDIDATE_Manifest_v1.0.3.csv \
  --out-dir reports/audits/retokenized \
  --retokenize
```

## 5. 데이터 무결성 규칙

- `data/candidate/` 안의 CSV는 입력본으로 취급하며 직접 덮어쓰지 않습니다.
- 수정 시 새 버전 파일을 만들고 `data/review/change_log.csv`에 변경 근거를 기록합니다.
- 자동 검사 통과는 의미적 검토 통과가 아닙니다.
- 검토 대기열의 기본 상태는 전부 `NOT_REVIEWED`입니다.
- 보고서의 상태는 `CANDIDATE_NEEDS_REVIEW` 또는 `AUDIT_FAIL`이며, 이 스크립트는 `FROZEN`을 출력하지 않습니다.
- 확인 분석의 표본 단위는 token이 아니라 `topic_id` 클러스터입니다.
- 확정 실험 실행 로그에 manifest SHA-256, 모델/토크나이저 식별자, 라이브러리 버전, 실행 설정을 저장합니다.

## 폴더 구성

- `data/candidate/`: 현재 후보 manifest
- `data/history/`: 과거 후보 버전(기록 보존용)
- `data/review/`: 사람 검토 대기열과 변경 이력
- `scripts/audit_manifest.py`: 결정론적 구조 감사기
- `notebooks/`: Colab에서 실행할 체크/감사 노트북
- `notebooks/archive/`: 기존 N02 노트북. 기준본 또는 확인 결과로 간주하지 말 것
- `reports/audits/`: 감사 산출물
- `data/history/VERSION_PROVENANCE.md`: 기존 버전·해시 및 과거 동결 보고서와의 차이 기록
- `AGENTS.md`, `.agent/rules/`: 에이전트 작업 원칙

## 체크섬

`CHECKSUMS.sha256`에는 workspace 파일별 SHA-256이 있습니다(체크섬 파일 자체는 목록에서 제외). 배포 ZIP의 체크섬은 ZIP 파일을 받은 뒤 별도로 계산할 수 있습니다.
