# 보관자 절차 — 받기·검사·분할·봉인

보관자는 원본, 분할 salt, 봉인 자료를 혼자 가진다. 가능하면 지금 검색 정책 개발(튜닝·실험·에이전트 세션)에 참여하지 않는 사람이 맡는다. 개발자가 맡을 수밖에 없으면 그 사실을 `roles.json`의 `keeper_involved_in_search_dev`에 적는다. 그때 상태 문서와 최종 보고에는 "봉인 보관자가 개발 참여자"라는 한계가 붙는다.

모든 명령은 저장소 루트에서 메인 체크아웃의 `venv/bin/python`으로 돌린다. 모델·검색·외부 호출은 없다. 화면에는 건수만 나온다.

```bash
T=experiments/collection_v11
K=<보관자 개인 드라이브의 폴더>     # Git 작업 트리·/tmp 안이면 도구가 거부한다
PY=venv/bin/python
```

## 0. 처음 한 번

| 할 일 | 명령 · 메모 |
|---|---|
| 보관자 폴더 만들기 | `$PY $T/collection_tool.py init --private-dir $K` |
| salt 만들기와 해시 약속 | `$PY $T/collection_tool.py commit-salt --private-dir $K` — 화면에 나온 `salt_sha256`만 PM에게 알린다(상태 문서에 실린다). salt 파일 자체는 누구에게도 보내지 않는다. 두 번째 실행은 거부된다 |
| 역할 기록 | `$T/packets/keeper/roles_template.json`을 `$K/roles.json`으로 복사해 코드로만 채운다. 이름↔코드 대응은 `$K/roster_private.csv`처럼 따로 둔다(도구는 읽지 않는다) |
| 확인자용 코퍼스 목록 | `$PY $T/collection_tool.py export-corpus-meta --out $K/packets/corpus_meta.csv` — 제목·가수·앨범·발매일만. **작성자에게는 주지 않는다**(목록을 보고 곡을 고르면 분포가 바뀐다) |

## 1. 작성자에게 보내기

### 1-0. 시범(2026-10-20 결정: 작성자 3명 × 10건, `general`)

- 작성자 3명에게 코드를 주고 `--rows 10 --track general`로 패킷을 만든다. 확인자는 그 3명이 아닌 사람으로 정한다.
- 시범에서는 한 사람 몫이 약 33%라 20% 기준을 넘는다. 이 기준은 **본 수집을 마친 시점의 누적 `general`**에 적용한다(`DATA_SPEC.md` §7). 비율을 맞추려고 받은 질의를 버리지 않는다. 본 수집에서 작성자를 늘리고 묶음 크기를 조절해 맞춘다.
- 시범 자료도 절차가 같으면 그대로 분할한다. 시범 뒤 안내·양식을 고치면, 고치기 전 자료와 고친 뒤 자료를 나눠 센다(`event --type note`로 고친 시점을 남긴다).
- 시범에서 볼 것: 형식 오류율, 질의당 걸린 시간(작성자에게 대략만 묻는다), 코퍼스 밖·확인 대기 비율. 품질 지표(Hit 등)는 보지 않는다.

### 1-1. 패킷 만들기

1. 사람마다 코드를 정한다(`P01`…). 본 수집에서는 누적 몫이 20%를 넘지 않게 묶음 크기를 정한다(10~25).
2. `$PY $T/collection_tool.py make-writer-packet --private-dir $K --code P03 --rows 20 --track general --out-dir $K/packets/P03`
3. 보낼 것: `writer_sheet_P03_….csv` · `declaration_P03.csv` · `$T/packets/writer/WRITER_GUIDE.md` · `$T/packets/CONSENT_PRIVACY.md`(빈칸을 PM이 채운 판). **그 밖의 파일·저장소 링크·기존 질의·개발 결과·실패 목록은 보내지 않는다.**
4. `diag_work`·`diag_sound` 진단 갈래는 그 작성자의 `general` 묶음을 받은 뒤에만 따로 부탁한다. 진단 갈래를 먼저 쓰면 일반 갈래 문장이 그쪽으로 기운다. 진단 갈래 안내는 `README.md` §4를 따른다.

## 2. 받기 — 받은 그대로 보존

```bash
$PY $T/collection_tool.py intake --private-dir $K --kind declaration --code P03 --file <받은 declaration>
$PY $T/collection_tool.py intake --private-dir $K --kind writer --code P03 --file <받은 writer_sheet>
$PY $T/collection_tool.py check --private-dir $K
```

- 받은 파일은 바이트 그대로 `$K/received/`에 읽기 전용으로 남고 해시가 기록된다. 원본 파일은 고치지 않는다.
- `check`가 오류를 내면 작성자에게 **형식만** 되묻는다(어느 열 값이 틀렸는지). 내용 힌트("단서를 더 넣어 달라")는 주지 않는다.
- 고친 표를 받을 때는 수정본으로 받는다. `--reason "<이유>" --editor <고친 사람 코드>`를 붙인다. 질의 원문이 바뀐 행은 `edit_confirmed_by_writer=yes`와 `edit_note`가 있어야 도구가 받는다. 최초본은 그대로 남는다.
- 같은 파일을 두 번 넣어도 기록은 늘지 않는다. 이미 받은 항목을 새 파일로 넣으면 거부된다.
- 문장에 다른 사람을 알아볼 정보가 있으면 가린 판을 수정본으로 받는다(`--reason "개인정보 가림"`). 작성자에게 알린다.

## 3. 정답 확인

```bash
$PY $T/collection_tool.py make-checker-sheet --private-dir $K --writer P03 --out $K/packets/checker_P03_r1.csv
```

- 확인자에게 보낼 것: 그 표 · `corpus_meta.csv` · `$T/packets/checker/CHECKER_GUIDE.md` · 동의 안내. 확인자 표는 작성자 한 명씩 만든다(참여 철회 때 그 사람 자료만 지우려고).
- 작성자 표·확인자 표 사본은 **보관자 폴더 안에만** 만들 수 있다(도구가 밖을 거부한다). 만든 사본은 기록된다.
- 사본을 밖으로 보낼 때마다 기록한다: `deliver --file packets/checker_P03_r1.csv --to <받은 사람 코드> --by <코드>`. policy_dev 내보내기를 개발 쪽에 줄 때도 `--to dev`로 기록한다. 봉인 내보내기는 도구가 거부한다. 이 기록이 있어야 철회 때 누구에게 삭제를 요청할지 알 수 있다.
- 작성자 본인이 확인자가 되지 않게 배정한다. 사람이 모자라 그렇게 되면 그대로 받는다. 도구가 `self_checked`로 표시한다.
- 받은 표: `intake --kind checker --code <확인자 코드> --file …` → `check`.

## 4. 분할 — 확인을 받은 직후, 결과를 보기 전

```bash
$PY $T/collection_tool.py split --private-dir $K
$PY $T/collection_tool.py export --private-dir $K
$PY $T/collection_tool.py status --private-dir $K --out $K/COLLECTION_STATUS.md
$PY $T/collection_tool.py verify --private-dir $K
```

- 묶음이 들어올 때마다 바로 나눈다. 한 번 배정된 항목은 옮겨지지 않는다.
- `split`이 "두 분할을 잇는다"며 멈추면(수정본이 원인), 보관자가 판단해 `quarantine --entry <번호> --reason … --by <코드>`로 그 항목을 양쪽에서 빼고 다시 돌린다. 어느 쪽 분할이 더 좋아 보이는지로 고르지 않는다. 항상 연결을 만든 쪽 항목을 뺀다.
- salt·비율은 바꿀 수 없다. 분할 결과가 마음에 들지 않아도 다시 뽑지 않는다.

## 5. 개발 쪽에 넘기는 것

| 넘김 | 넘기지 않음 |
|---|---|
| `$K/exports/policy_dev_rNNN_*.csv`(개발 자료) | `sealed_eval_*` · `salt.txt` · `received/` · `work/merged.csv` · `roster_private.csv` · `roles.json` |
| `COLLECTION_STATUS.md`(익명 집계) | 봉인 문장·정답·질의별 결과 |

- policy_dev 원문도 공개 저장소에 커밋하지 않는다(동의 안내 약속). 개발 세션은 Git 밖 경로에서 읽는다.
- 개발 쪽이 policy_dev로 처음 모델을 돌리면 그 시점을 기록한다: `event --type model_outputs_released --scope policy_dev --by <코드> --note "<무슨 실행>"`. 그 뒤로 policy_dev는 개발 자료다. 독립 최종 평가 자료라고 부르지 않는다.
- 데이터 고정 시점(`data_frozen`)은 `export`가 자동으로 남긴다. 두 시점은 따로 기록된다.

## 6. 봉인 평가 — 구성과 채택 기준을 고정한 뒤 한 번

1. PM이 비교할 구성(커밋·환경·색인 지문)과 채택 기준(지표 우선순위·판정 규칙 숫자)을 문서로 고정한다 → `event --type criteria_fixed --file <그 문서> --configs "A,B" --by <코드>`.
2. `event --type sealed_opened --by <코드>`. 두 번째 열기는 거부된다. 이 뒤로는 이 회차에 새 배정을 하지 않는다.
3. 평가는 보관자가 직접 돌리거나, 질의 원문·질의별 순위를 출력하지 않는 하네스로 돌린다. 개발 세션에는 집계·신뢰구간·갈린 질의 수(얻음/잃음)만 넘긴다. 질의별 실패 분석을 하는 순간 그 세트는 개발용으로 강등되고, 새 봉인 세트가 필요하다.
4. 같은 봉인 세트로 구성을 다시 고르지 않는다.

## 7. 참여 철회

```bash
$PY $T/collection_tool.py withdraw --private-dir $K --person P03 --by <코드> --reason "본인 요청"
```

보관자 폴더 안에서 다음을 지운다.

- 그 사람이 낸 표와 그 사람 문장을 담은 확인 표
- 그 사람 항목이 든 내보내기
- 그 사람 몫의 작성자 패킷·확인자 표 사본
- 파생 작업 파일(`work/` 전부)

`work/`는 다음 `build`·`split`이 철회자 없이 다시 만든다. 지운 파일 이름과 시각만 기록에 남는다. 분할 장부에는 항목 번호만 남는다. 다음 `export`는 그 항목 없이 새 판을 만든다. 확인자로 남긴 판단까지 지우려면 `--include-checker-work`를 붙인다.

**밖으로 보낸 사본**(`deliver`로 기록한 확인자 표·policy_dev 내보내기)은 도구가 지울 수 없다. 그래서 삭제 요청(`DR-001`…)으로 남는다.

1. 받은 사람에게 삭제를 요청한다.
2. 지웠다는 회신을 받으면 `confirm-deletion --request DR-001 --by <코드> --note "<누가 언제 어떻게 지웠다고 회신>"`으로 기록한다.

대기 중인 요청 수는 상태 문서에 나온다. 기록하지 않고 보낸 사본은 도구가 모른다. 그러니 보낼 때마다 `deliver`를 빠뜨리지 않는다.

## 8. 하지 않는 것

- 받은 문장을 다듬거나 단서를 보태지 않는다. 내용 힌트를 주며 되묻지 않는다.
- 수집 자료를 보고 코퍼스·색인·프롬프트를 바꾸자고 제안하지 않는다.
- 봉인 자료를 에이전트 세션, `/private/tmp`, 저장소, 팀 공유 폴더에 복사하지 않는다.
- 기존 평가 질의 원문을 참여자에게 보여 주지 않는다. 중복 검사는 도구가 보관자 쪽에서만 하고, 결과는 건수로만 나온다.
