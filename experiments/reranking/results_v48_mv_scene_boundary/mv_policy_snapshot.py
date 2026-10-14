"""MV 장면 판정 스냅샷 — `rerank_policy`의 순수 함수만 부른다(모델·DB·분석 API 호출 없음).

    venv/bin/python experiments/reranking/results_v48_mv_scene_boundary/mv_policy_snapshot.py <out.json>

- 대표 오탐·실제 장면·범위 밖 사례의 판정(`is_specific_mv_scene_query`)과 표시 구간(`mv_scene_phrases`)
- dev 176건(v06 57 · v09 59 · v10 60 — `eval_queries_v*.csv`의 dev 분할, 분석 캐시 원문과 같은지 확인)의 판정·표시 구간

봉인 test 분할은 읽지 않는다. 수정 전후 비교는 수정 전 커밋에서 한 번, 수정 후 커밋에서 한 번 돌려 JSON을 견준다
(`dev_policy_before_after.csv` · `probe_cases_before_after.csv`가 그 결과다). 다른 체크아웃의 판정을 재려면
`VAGUEFINDER_POLICY_ROOT=<그 체크아웃>`을 준다(dev 질의·캐시는 이 체크아웃에서 읽는다).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POLICY_ROOT = Path(os.environ.get("VAGUEFINDER_POLICY_ROOT", ROOT)).resolve()
sys.path.insert(0, str(POLICY_ROOT))

from src.retrieval.rerank_policy import is_specific_mv_scene_query, mv_scene_phrases, mv_scene_spans  # noqa: E402

FALSE_POSITIVES = [
    # §2-29 ① — 아이(들)·소년·소녀의 낱말 경계
    "뮤비에서 춤추는 아이돌 노래", "뮤비에서 춤추는 아이유 노래", "뮤비에서 춤추는 아이브 노래", "뮤비에서 웃는 아이린이 예뻤던 노래",
    "뮤비에서 춤추는 소녀시대 노래", "뮤비에서 춤추는 방탄소년단 노래", "뮤비에서 춤추는 소녀감성 노래", "뮤비에서 아이스크림 먹는 노래",
    "뮤비에서 춤추는 남자 아이돌 노래", "뮤비에서 춤추는 여자 아이돌 노래", "뮤비에서 춤추는 아이돌노래", "뮤비에서 춤추는 아이돌의 노래",
    "뮤비에서 춤추는 아이돌, 제목은 몰라", "뮤비에서 춤추는 아이돌이 부른 노래", "뮤비에서 춤추는 아이돌그룹 노래",
    # §2-29 ② — 남자/여자 뒤 화제 전환
    "뮤비에서 춤추는 여자 목소리 노래", "뮤비에서 춤추는 남자 보컬 노래", "뮤비에서 우는 여자 음색이 좋은 노래", "뮤비에서 춤추는 여자목소리 노래",
    "뮤비에서 춤추는 남자보컬 노래", "뮤비에서 춤추는 여자 목소리, 발라드야", "뮤비에서 춤추는 여자 목소리였어", "뮤비에서 걷는 남자 목소리가 좋은 노래",
    # PR #61 자동 리뷰 [참고] — 관형격·복수도 같은 가수 화제
    "뮤비에서 춤추는 여자의 목소리가 좋은 노래", "뮤비에서 춤추는 여자들 목소리 노래", "뮤비에서 우는 남자의 음색이 좋은 노래",
]

TRUE_SCENES = [
    # 리뷰 지적(10/14 1·2차) — 겹낱말 장면(노래 바로 앞 포함) · 인물 낱말의 서술격 활용. 처음 시안들이 잃었던 것
    "뮤비에서 벚꽃이 흩날리는 노래", "뮤비에서 벚꽃 흩날리는 노래", "뮤비에서 흩날리는 벚꽃 노래", "뮤비에서 날아다니는 종이비행기 노래",
    "뮤비에서 춤추는 어린아이 노래", "뮤비에 밤하늘이 나와", "뮤비에서 어린아이가 춤추는 노래",
    "뮤비에서 고등학교에서 춤추는 노래", "뮤비에서 청소년들이 옥상에서 춤추는 노래", "뮤비에서 구급차가 달려", "뮤비에서 손거울을 들고 걷는 노래",
    "뮤비에서 춤추는 소녀였어", "뮤비에서 춤추는 아이야", "뮤비에서 춤추는 소년인데 제목을 몰라", "뮤비에서 우는 아이들이었어", "뮤비에서 춤추는 소녀예요",
    "뮤비에서 웃고 있는 소년이다", "뮤비에서 춤추는 아이거든", "뮤비에서 웃는 아이죠", "뮤비에서 춤추는 소년이네요", "뮤비에서 춤추는 아이더라",
    "뮤비에서 우는 소녀일까", "뮤비에서 춤추는 아이잖아",
    # PR #61 자동 리뷰 [권장] — 조사 뒤에 띄지 않고 붙여 쓴 서술어·구조어
    "뮤비에서 아이가춤추는 노래", "뮤비에서 소녀가울고 있어", "뮤비에서 아이들이랑춤추는 노래", "뮤비에서 소년이나오는 노래",
    # 리뷰 3차(10/14) — 명사에 바로 붙는 '요' · 복합 조사 · 결합형
    "뮤비에서 춤추는 아이요", "뮤비에서 울던 소녀요", "뮤비에서 소년에게서 빛이 나와", "뮤비에서 춤추는 아이도요", "뮤비에서 소녀께서 춤추는 노래",
    "뮤비에서 소년들에게서는 빛이 나와", "뮤비에서 춤추는 아이들한테서도요",
    "뮤비에서 아이가 춤추는 노래", "뮤비에 소녀가 울고 있는 장면이 있어", "뮤비에서 아이들이랑 춤추는 노래", "뮤비에서 아이들이 풍선 들고 뛰어다녀",
    "뮤비에서 소년이 달리는 노래", "뮤비에서 소년과 소녀가 걷는 노래", "뮤비에서 아이 혼자 걷는 노래", "뮤비에서 아이가 춤추는 노래, 제목은 몰라",
    "뮤비에서 아이가 춤춰", "뮤비에서 춤추는 아이가 나와", "뮤비에서 아이들 뛰어다니는 노래", "뮤비에서 소녀 혼자 우는 노래",
    "뮤비에서 아이한테 풍선 주는 노래",        # '주는'은 시각 서술어 목록에 없다 — 수정 전에도 비대상(이번 범위 아님)
    "뮤비에서 아이처럼 뛰어다니는 노래",
    "뮤비에서 여자가 춤추는 노래", "뮤비에서 의자를 던져, 목소리는 여자였어", "뮤비에서 여자가 울고 있어 목소리는 남자였어",
    "뮤비에서 여자가 춤추는데 보컬은 남자야", "뮤비에서 춤추는 사람은 여자였어", "뮤비에서 남자가 달리고 여자가 울어", "뮤비에서 춤추는 여자가 나와",
    "뮤비에서 바다가 나오는 노래", "뮤비에 꽃나무가 가득해", "뮤비에서 교복입고 춤추는 노래", "뮤비에서 눈밭을 달리는 노래",
    "뮤비에서 나비가 날아다니는 노래", "뮤비에서 버스 타고 어디 가", "뮤비에서 상자를 던지는 노래", "뮤비에서 별이 쏟아지는 노래",
    "뮤비에서 태양이 떠오르는 노래", "뮤비에서 유리 상자 안에 갇혀 있어", "뮤비에서 의자가 하늘에 떠 있어 여자 목소리였어",
    "뮤비에서 의자가 하늘에 떠 있는 춤추는 여자 목소리 노래",
]

# §2-29 ③(보류)·④·⑤ — 이번에 고치지 않았다. 기대값을 고정하지 않고 기록만 한다
OUT_OF_SCOPE = [
    # ③ 명사로 끝나는 가수 이름 — "흩날리는 벚꽃 노래"와 문형이 같아 문장만으로는 가를 수 없다(이름 사전 필요)
    "뮤비에서 춤추는 잔나비 노래", "뮤비에서 춤추는 유리상자 노래", "뮤비에서 춤추는 임한별 노래", "뮤비에서 춤추는 잠골버스 노래",
    "뮤비에서 춤추는 서울달 노래", "뮤비에서 춤추는 새별 노래", "뮤비에서 춤추는 잔나비노래", "뮤비에서 춤추는 잔나비의 노래",
    "뮤비에서 잔나비 노래 부르는 곡", "뮤비에서 춤추는 잔나비 곡이야", "뮤비에서 춤추는 아이오아이 노래",
    "뮤비에서 잔나비가 춤추는 노래", "뮤비에서 유리상자가 노래 부르는 곡", "뮤비에서 임한별이 우는 노래", "뮤비에서 아이오아이가 춤추는 노래",
    "뮤비에서 춤추는 잔나비, 제목은 몰라", "뮤비에서 춤추는 잔나비",
    # ④·⑤
    "뮤비에서 춤추는 태양 노래", "뮤비에서 춤추는 별 노래", "뮤비에서 태양이 춤추는 노래", "뮤비에서 바다가 노래 부르는 곡",
    "뮤비에서 춤추는 버스커 버스커 노래", "뮤비에서 춤추는 여자친구 노래", "뮤비에서 춤추는 칼라파워 노래", "뮤비에서 춤추는 아이들 노래",
    "뮤비에서 춤추는 여자아이들 노래", "뮤비에서 춤추는 노르웨이 숲 노래",
]


def snap(query: str) -> dict:
    return {"target": is_specific_mv_scene_query(query), "spans": mv_scene_spans(query), "phrases": mv_scene_phrases(query)}


def main(out: Path) -> None:
    result = {
        "false_positives": {q: snap(q) for q in FALSE_POSITIVES},
        "true_scenes": {q: snap(q) for q in TRUE_SCENES},
        "out_of_scope": {q: snap(q) for q in OUT_OF_SCOPE},
        "dev": {},
    }
    exp = ROOT / "experiments" / "reranking"
    for version in ("v06", "v09", "v10"):
        cache = json.loads((exp / f"analysis_cache_{version}_dev.json").read_text(encoding="utf-8"))["entries"]
        with (exp / f"eval_queries_{version}.csv").open(encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("split", "").lower() == "dev"]
        for row in rows:
            qid, query = row["query_id"], row["query"]
            entry = snap(query)
            entry["cache_query_matches"] = cache.get(qid, {}).get("query") == query
            entry["set"] = version
            result["dev"][qid] = {"query": query, **entry}
    targets = [k for k, v in result["dev"].items() if v["target"]]
    mismatch = [k for k, v in result["dev"].items() if not v["cache_query_matches"]]
    result["dev_summary"] = {"count": len(result["dev"]), "targets": targets, "cache_query_mismatch": mismatch}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"dev={len(result['dev'])} targets={len(targets)} cache_mismatch={len(mismatch)}")
    for group in ("false_positives", "true_scenes", "out_of_scope"):
        print(f"--- {group}")
        for query, entry in result[group].items():
            print(f"  {'대상 ' if entry['target'] else '비대상'} | {query} | {entry['phrases']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mv_policy_snapshot.json"))
