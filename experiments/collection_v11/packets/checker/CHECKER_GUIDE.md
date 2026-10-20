# 정답 확인 — 확인자 안내

작성자가 기억으로 쓴 검색 문장마다 **작성자가 떠올린 노래가 무엇인지**, 그 노래가 우리 노래 목록(코퍼스)에 있는지 확인합니다. 시작하기 전에 `CONSENT_PRIVACY.md`를 읽고 `declaration_<코드>.csv`를 채워 주세요.

## 받은 파일

| 파일 | 내용 |
|---|---|
| `checker_<작성자코드>_….csv` | 작성자 한 명의 문장과, 작성자가 생각한 노래(있으면). 오른쪽 빈 열을 채웁니다 |
| `corpus_meta.csv` | 코퍼스 노래 목록: `song_id` · 제목 · 가수 · 앨범 · 발매일만 |

이 표에는 검색 결과·후보·점수가 없습니다. 일부러 뺐습니다.

## 지킬 것

- **우리 검색 서비스에 문장을 넣어 보지 않습니다.** 검색 결과·실험 결과·개발팀의 실패 목록도 보지 않습니다. 이미 본 문장이면 `saw_search_results`에 `yes`라고 적습니다.
- AI 챗봇에게 노래를 맞혀 달라고 하지 않습니다. 정답은 작성자의 기억입니다.
- 문장(`query`)은 고치지 않습니다. 이상한 점은 `checker_notes`에 적습니다.
- 확인이 안 되는 행도 지우지 않습니다. `pending`이나 `undeterminable`로 적습니다.

## 정답 정하는 법

1. **정답은 작성자가 떠올린 노래입니다.** 문장에 더 잘 맞는 다른 노래가 있어도, 작성자의 노래가 정답입니다. 작성자가 단서를 잘못 기억했으면(성별·시기·장르 등) 정답은 그대로 두고 `wrong_clue_types`와 `wrong_clue_note`에 적습니다.
2. 작성자가 노래를 적었으면 `corpus_meta.csv`에서 제목·가수로 찾습니다. 멜론·유튜브·가사 사이트에서 노래를 확인해도 됩니다(`external_source`·`listened`).
3. 작성자가 노래를 몰랐으면, 작성자에게 **어떤 노래였는지만** 물어봅니다. 후보 목록을 보여 주거나 "혹시 이 노래?"라고 유도하지 않습니다(`writer_interview`). 답을 기다리는 동안은 `pending`입니다.
4. 같은 노래의 다른 판(원곡·리메이크·라이브·리마스터)이 코퍼스에 여럿 있으면, 원곡을 맨 앞에 두고 모두 적습니다. 이때 `multiple_answers`=`versions`입니다.
5. 작성자도 둘 이상 중에서 못 가르면 `ambiguous`로 적고, 후보를 모두 적습니다(`multiple_answers`=`distinct`).

## 열 채우기

| 열 | 값 |
|---|---|
| `answer_status` | `confirmed` · `ambiguous` · `undeterminable` · `pending` |
| `answer_ids` | `corpus_meta.csv`의 `song_id`. 여럿이면 `\|`로(`30486509\|12345678`) |
| `multiple_answers` | `no` · `versions` · `distinct` |
| `in_corpus` | `yes`(정답이 목록에 있음) · `no`(없음) · `partial`(일부 판만 있음) · `unknown`(pending·undeterminable만) |
| `ooc_title` · `ooc_artist` | 목록에 없는 정답 노래의 제목·가수 |
| `evidence` | `;`로 여럿: `writer_named`(작성자가 적음) `metadata`(목록에서 찾음) `cover` `listened` `lyrics_source` `external_source` `writer_interview` `other` |
| `confidence` | `high` · `medium` · `low` |
| `wrong_clue_types` | `none` 또는 `;`로: `gender` `era` `genre` `group_solo` `instrument` `voice` `lyrics` `work` `cover` `mv` `title_shape` `other` |
| `wrong_clue_note` | 무엇이 어떻게 달랐는지 한 줄 |
| `work_check_result` | 작성자가 작품명을 적었을 때만: 그 작품이 실제로 있고 문장이 그 작품을 가리키면 `same`, 아니면 `different`, 모르면 `unsure` |
| `work_in_song_data` | 그 노래가 그 작품에 쓰였다는 것을 목록의 앨범명·발매 정보로 확인했으면 `yes`, 아니면 `no`, 모르면 `unknown` |
| `checker_code` · `checked_at` | 내 코드 · 확인한 시각 `2026-10-22 15:30` |
| `saw_search_results` | 이 문장의 검색 결과를 본 적이 있으면 `yes` |
| `checker_notes` | 자유 |

- `confirmed`이면 `confidence`와 `evidence`가 꼭 있어야 합니다.
- 내가 쓴 문장을 내가 확인하게 되면 그대로 적습니다. 도구가 자동으로 표시하고, 독립 확인과 따로 셉니다.
- 다 채우면 보관자에게 보냅니다. 나중에 고칠 때는 보관자에게 알리면 새 판으로 남깁니다.
