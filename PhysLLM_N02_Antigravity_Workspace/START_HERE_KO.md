# PhysLLM N02 — 여기서 시작하세요

## A. Antigravity IDE 열기

1. `PhysLLM_N02_Antigravity_Workspace.zip`을 내려받아 로컬에서 압축 해제합니다.
2. Antigravity IDE에서 `PhysLLM_N02.code-workspace` 또는 폴더 자체를 엽니다.
3. Extensions에서 `Google.colab`, `ms-toolsai.jupyter`, `ms-python.python`을 설치/확인합니다.
4. `notebooks/00_colab_runtime_check.ipynb`를 열고 커널 선택 메뉴에서 `Colab → Auto Connect`를 선택합니다.

Google Colab 확장의 공식 문서에서 안내하는 연결 절차는 VS Code 기준입니다. Antigravity에서 확장이 목록에 없거나 연결이 실패하면 같은 노트북을 Google Colab 웹 또는 VS Code에서 열어 같은 순서로 실행하세요.

## B. 파일 공유 연결

Colab은 원격 런타임이므로 로컬 폴더가 자동으로 공유된다고 가정하지 않습니다.

- **권장:** Google Drive for desktop 동기화 폴더 안에 workspace를 두고, 그 폴더를 Antigravity에서 엽니다.
- **대안:** 압축 해제한 폴더를 Drive의 `My Drive`에 업로드합니다. 코드 파일을 IDE에서 수정한 뒤 실행하려면 수정한 파일이 Drive에 동기화/업로드되었는지 확인합니다.

목표 Drive 경로:
`MyDrive/PhysLLM_N02_Antigravity_Workspace/`

## C. 실험 시작 전

1. `00_colab_runtime_check.ipynb` 실행 — 런타임 연결 확인
2. `01_manifest_integrity_audit.ipynb` 실행 — CPU 기준 후보 manifest 감사
3. `02_review_queue_overview.ipynb` 실행 — 사람 검토 대기열 점검
4. `data/review/HUMAN_REVIEW_GUIDE_KO.md`에 따라 400행을 검토하고, 근거·검토자·UTC 시간을 남깁니다.

현재 상태는 `CANDIDATE_NEEDS_REVIEW`입니다. 기본 구조 검사는 통과했지만, 의미 검토와 독립 토크나이저 재측정이 완료되지 않았습니다. 아직 확인 실험용으로 동결하지 않습니다.

## D. IDE 안의 빠른 명령

- `Terminal → Run Task` 또는 명령 팔레트에서 `Tasks: Run Task`를 열고 `N02: Audit candidate manifest (local CPU)` 실행
- 테스트 실행: `N02: Run workspace tests`
- 터미널에서 직접 실행하려면 `README_KO.md`를 참고하세요.
