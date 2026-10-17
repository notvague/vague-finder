"""new_data_check 자체 검사 — 손으로 만든 입력. 코퍼스·기존 질의는 가짜. 모델·검색 없음."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import new_data_check as nd  # noqa: E402

fails = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def row(tid, q, pid="", track="general", mem="own_memory", before="yes"):
    return {"tmp_id": tid, "writer_id": "w", "memory_source": mem, "written_before_lookup": before, "query": q,
            "track": track, "tier": "vague", "positive_id": pid, "corpus_status": ""}


corpus = {"1", "2", "3", "4", "5"}
existing = [{"text": "비 오는 날 듣던 남자 발라드인데 피아노가 예뻤어", "ids": {"9"}},
            {"text": "체육대회 때 틀던 신나는 댄스곡", "ids": {"5"}}]
rows = [
    row("a1", "고등학교 축제 때 다 같이 떼창하던 밴드 노래", "1"),
    row("a2", "축제에서 떼창하던 그 밴드 곡, 후렴이 구호 같았음", "1"),               # 같은 곡 → 같은 분할
    row("a3", "비 오는 날 듣던 남자 발라드인데 피아노가 예뻤어요", "2"),             # 기존 문장과 거의 같음
    row("a4", "체육대회 노래였는데 여자 그룹", "5"),                                 # 기존 정답 곡과 겹침
    row("a5", "엄마가 차에서 틀던 트로트", "77"),                                    # 코퍼스 밖
    row("a6", "그 노래", ""),                                                        # 정답 미확인 + 짧음
    row("a7", "합성: 도입부 사이렌 소리 나는 힙합", "3", track="diag_sound", mem="synthetic"),
    row("a8", "드라마 보다가 들은 노래 같은데 기억이 흐릿함", "4", track="diag_work", before="no"),
]
rep = nd.check(rows, corpus, existing, "s" * 20, 0.6)
flags = {r["tmp_id"]: set(filter(None, r["flags"].split("|"))) for r in rows}
split = {r["tmp_id"]: r["split"] for r in rows}
check("같은 곡 두 질의는 같은 분할", split["a1"] == split["a2"] and split["a1"] in ("sealed", "policy_dev"))
check("기존 질의와 비슷한 문장 → near_duplicate·제외", "near_duplicate" in flags["a3"] and split["a3"] == "excluded")
check("기존 정답 곡과 겹침 → song_overlap·제외", "song_overlap" in flags["a4"] and split["a4"] == "excluded")
check("코퍼스 밖 → out_of_corpus·제외", "out_of_corpus" in flags["a5"] and split["a5"] == "excluded")
check("정답 미확인·짧은 질의 표시(제외는 아님)", {"unresolved", "short"} <= flags["a6"] and split["a6"] != "excluded")
check("합성 자료 → synthetic·제외(독립 증거 아님)", "synthetic" in flags["a7"] and split["a7"] == "excluded")
check("정답 본 뒤 작성 표시", "written_after_lookup" in flags["a8"])
check("문제 없음", rep["problems"] == [])
rep2 = nd.check([dict(r) for r in rows], corpus, existing, "s" * 20, 0.6)
check("같은 salt면 같은 분할(결정적)", rep2["split_counts"] == rep["split_counts"])
bad = [row("b1", "x", "1", track="weird"), row("b1", "y", "2", mem="guess")]
rep3 = nd.check(bad, corpus, existing, "s" * 20, 0.6)
check("잘못된 track·memory_source·tmp_id 중복 거부", len(rep3["problems"]) == 3)
print(f"FAILURES {len(fails)}")
sys.exit(1 if fails else 0)
