"""실패 경로를 **화면에서** 확인한다 — 실제 브라우저로 실제 화면을 돌린다.

`demo_faults.py`가 장애를 주입하고, 여기서는 그 앞에 브라우저를 세워 화면이 무엇을
말하는지 본다. 응답 JSON만 보면 알 수 없는 것들이 있다 — 로딩이 풀렸는지, 버튼이
눌리는 상태로 돌아왔는지, 늦게 온 응답이 새 결과를 덮었는지.

확인 기준(항목마다 같다): **오류 안내 → 로딩 해제 → 재시도 → 정상 복구**

    venv/bin/python -m src.backend.check_demo_faults            # 서버까지 띄운다
    venv/bin/python -m src.backend.check_demo_faults --url http://127.0.0.1:8010
    venv/bin/python -m src.backend.check_demo_faults --headed   # 눈으로 볼 때

크롤러가 쓰는 selenium을 그대로 쓴다(requirements에 이미 있다). Chrome이 필요하다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

DEFAULT_URL = "http://127.0.0.1:8010"
# 결과가 0건일 때 화면이 내는 문구. 다른 상황이 같은 말을 하면 구분이 안 된다.
EMPTY_MESSAGE = "일치하는 곡이 없습니다."
QUERY = "앨범이 파란 잔잔한 노래"


# ---------------------------------------------------------------------------
# 화면 읽기 — map.js가 실제로 그린 것만 본다
# ---------------------------------------------------------------------------

_SNAPSHOT_JS = r"""
const t = (el) => (el && !el.hidden ? (el.textContent || "").trim() : "");
const status = document.getElementById("status");
const runNote = document.getElementById("runNote");
const clarify = document.getElementById("clarifyBox");
return {
  status: t(status),
  statusWarn: !!(status && status.classList.contains("warn")),
  runNote: t(runNote),
  runNoteWarn: !!(runNote && runNote.classList.contains("warn")),
  clarify: t(clarify),
  results: [...document.querySelectorAll("#resultList .resultitem")]
      .map((li) => (li.querySelector(".resultsong") || {}).textContent || ""),
  hits: document.querySelectorAll(".song.hit").length,
  buttons: [...document.querySelectorAll("#clarifyBox button")]
      .map((b) => ({ text: (b.textContent || "").trim(), disabled: b.disabled })),
  explainToggles: document.querySelectorAll("#resultList .extoggle").length,
  panelMode: (typeof panelMode === "undefined" ? "?" : panelMode),
};
"""


@dataclass
class Case:
    """한 상황의 관찰 결과. 판단은 사람이 읽는 칸에 남긴다."""

    name: str
    note: str = ""
    observations: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def check(self, ok: bool, claim: str) -> None:
        (self.observations if ok else self.failures).append(
            ("OK   " if ok else "안 됨 ") + claim
        )

    def see(self, text: str) -> None:
        self.observations.append("     " + text)


class Screen:
    def __init__(self, driver, base: str, shots: Optional[Path]) -> None:
        self.d = driver
        self.base = base
        self.shots = shots

    # -- 장애 스위치 --------------------------------------------------
    def fault(self, mode: str, **kw: Any) -> None:
        body = json.dumps({"mode": mode, **kw}).encode()
        req = urllib.request.Request(
            f"{self.base}/__fault", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()

    def release(self) -> None:
        req = urllib.request.Request(f"{self.base}/__fault/release", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()

    # -- 조작 ---------------------------------------------------------
    # 검색 응답 원본을 붙잡아 둔다. 근거 패널에는 경로 **가중치**가 없어서,
    # "서버가 말한 것"과 "화면이 보여준 것"을 맞춰 보려면 응답이 있어야 한다.
    _HOOK_JS = """
    (() => {
      if (window.__vfHooked) return;
      window.__vfHooked = true;
      const real = window.fetch;
      window.fetch = async (...args) => {
        const res = await real(...args);
        const url = typeof args[0] === "string" ? args[0] : (args[0] || {}).url || "";
        if (url.includes("/api/v1/search")) {
          window.__vfStatus = res.status;
          try { window.__vfLast = await res.clone().json(); } catch { window.__vfLast = null; }
        }
        return res;
      };
    })();
    """

    def open(self) -> None:
        self.d.get(f"{self.base}/map")
        self.d.execute_script(self._HOOK_JS)
        self.wait(lambda s: True, timeout=10)

    def search(self, query: str = QUERY) -> None:
        self.d.execute_script(
            "const i = document.getElementById('searchInput');"
            "i.value = arguments[0];"
            "document.getElementById('searchForm').requestSubmit();",
            query,
        )

    def click(self, text: str) -> bool:
        """재질문 영역의 버튼을 문구로 누른다. 없으면 False."""
        return bool(self.d.execute_script(
            "const b = [...document.querySelectorAll('#clarifyBox button')]"
            "  .find((x) => (x.textContent || '').trim().startsWith(arguments[0]));"
            "if (!b || b.disabled) return false; b.click(); return true;",
            text,
        ))

    def esc(self) -> None:
        self.d.execute_script(
            "document.dispatchEvent(new KeyboardEvent('keydown',"
            "{key:'Escape',bubbles:true}));"
        )

    # -- 관찰 ---------------------------------------------------------
    def snap(self) -> Dict[str, Any]:
        return self.d.execute_script(_SNAPSHOT_JS)

    def wait(self, pred: Callable[[Dict[str, Any]], bool], timeout: float = 15.0):
        """조건이 참이 될 때까지. 시간이 지나면 마지막 상태를 그대로 돌려준다."""
        deadline = time.time() + timeout
        state = self.snap()
        while time.time() < deadline:
            if pred(state):
                return state
            time.sleep(0.15)
            state = self.snap()
        return state

    def settled(self, timeout: float = 20.0) -> Dict[str, Any]:
        """검색이 끝난 상태 — 결과가 있든 안내 문구가 떴든, 로딩만 아니면 된다."""
        return self.wait(
            lambda s: s["panelMode"] != "loading" and s["status"] != "검색 중…",
            timeout=timeout,
        )

    def shot(self, name: str) -> None:
        if self.shots is None:
            return
        self.shots.mkdir(parents=True, exist_ok=True)
        self.d.save_screenshot(str(self.shots / f"{name}.png"))


# ---------------------------------------------------------------------------
# 상황들
# ---------------------------------------------------------------------------

def case_quota(s: Screen) -> Case:
    """Gemini 쿼터 초과 — 규칙 폴백으로 검색은 계속된다."""
    c = Case("1. Gemini 쿼터 초과 (규칙 폴백)")
    s.fault("analysis_quota")
    s.open()
    s.search()
    st = s.settled()
    s.shot("1_quota")
    c.check(len(st["results"]) > 0, f"결과가 뜬다 ({len(st['results'])}곡)")
    c.check(st["panelMode"] != "loading", "로딩이 풀렸다")
    c.see(f"상태줄: {st['status'] or '(없음)'}")
    c.see(f"실행 기록: {st['runNote'] or '(없음)'}")
    x = _explain(s)
    c.see(f"응답의 analysis.confidence = {x['confidence']}")
    c.check(
        "분석" in st["status"] + st["runNote"] or "폴백" in st["status"] + st["runNote"],
        "질의 분석이 폴백이었다는 것을 화면이 말한다",
    )
    # 쿼터가 풀린 뒤 같은 화면에서 다시 검색하면 정상으로 돌아오는가
    s.fault("ok")
    s.search()
    st = s.settled()
    c.check(len(st["results"]) > 0, "쿼터 복구 후 재검색이 정상 동작한다")
    return c


def case_hang(s: Screen) -> Case:
    """Gemini가 응답하지 않음 — 로딩이 끝나는가, 빠져나갈 수 있는가."""
    c = Case("2. Gemini 시간 초과 (응답 없음)")
    s.fault("analysis_hang", seconds=600)
    s.open()
    s.search()
    time.sleep(3)
    st = s.snap()
    s.shot("2_hang_waiting")
    c.see(f"3초 뒤 상태줄: {st['status'] or '(없음)'}")
    c.check(
        st["status"] == "검색 중…",
        "기다리는 동안 '검색 중…'이 떠 있다",
    )
    # 더 기다려도 스스로 끝나는가 — 화면에 제한 시간이 있는지 본다.
    # 값은 페이지에서 읽는다. 여기 숫자를 따로 적으면 둘이 갈린다.
    limit = s.d.execute_script(
        "return typeof SEARCH_TIMEOUT_MS === 'undefined' ? null : SEARCH_TIMEOUT_MS;"
    )
    c.see(f"화면의 검색 제한 시간: {limit if limit else '(없음)'}")
    budget = (limit / 1000 + 6) if limit else 15
    st = s.wait(lambda x: x["status"] != "검색 중…", timeout=budget)
    c.check(
        st["status"] != "검색 중…",
        f"요청이 스스로 끝난다 ({budget:.0f}초 안에)",
    )
    c.see(f"끝났을 때 상태줄: {st['status'] or '(없음)'}")
    c.see(f"끝났을 때 실행 기록: {st['runNote'] or '(없음)'}")
    # 끝나는 길이 둘이다. 서버가 먼저 예산을 써 규칙 분석으로 내려오면 결과가
    # **정상 응답으로** 오고(그때는 실행 기록이 폴백을 말한다), 서버까지 막히면
    # 화면 제한이 걸려 상태줄이 이유를 말한다. 어느 쪽이든 **왜 이런 결과인지**는
    # 화면에 남아야 한다.
    said = st["status"] + st["runNote"]
    c.check(
        ("않" in said and "검색" in said) or "분석" in said,
        "왜 이런 결과인지 화면이 말한다",
    )
    # 사용자가 빠져나갈 길은 있는가
    s.esc()
    st = s.snap()
    c.check(st["status"] == "", "Esc로 검색을 취소할 수 있다")
    # 취소 후 정상 복구
    s.release()
    s.fault("ok")
    s.search()
    st = s.settled()
    s.shot("2_hang_recovered")
    c.check(len(st["results"]) > 0, "취소 후 재검색이 정상 동작한다")
    return c


def case_path_fail(s: Screen) -> Case:
    """검색 경로 하나가 죽음 — 설명이 실제 처리 상태에 맞는가."""
    c = Case("3. 검색 경로 실패 (이미지)")
    s.fault("ok")
    s.open()
    s.search()
    s.settled()
    healthy = _explain(s)

    s.fault("path_image_fail")
    s.search()
    st = s.settled()
    s.shot("3_path_fail")
    broken = _explain(s)

    c.check(len(st["results"]) > 0, f"남은 경로로 결과가 뜬다 ({len(st['results'])}곡)")
    c.see(f"정상일 때 1위의 경로: {healthy['serverPaths']}")
    c.see(f"이미지 실패 시 1위의 경로: {broken['serverPaths']}")
    c.see(f"응답이 말하는 경로 가중치: {broken['weights']}")
    image_weight = broken["weights"].get("image", 0)
    has_image_row = any("이미지" in p for p in broken["serverPaths"])
    c.see(f"실행 기록: {st['runNote'] or '(없음)'}")
    c.check(
        not (image_weight > 0 and not has_image_row) or "이미지" in st["runNote"],
        "가중치만 남고 기여가 사라진 경로를 화면이 설명한다",
    )
    if image_weight > 0 and not has_image_row and "이미지" not in st["runNote"]:
        c.see(
            f"가중치는 image={image_weight}로 남아 있는데 기여는 한 줄도 없다 — "
            "화면만 보면 '이미지 경로가 돌았지만 이 곡엔 기여가 없었다'로 읽힌다"
        )
    c.check(st["runNoteWarn"], "경고 표시가 붙는다")
    return c


def case_rerank_fail(s: Screen) -> Case:
    """Cross-Encoder가 죽음 — 검색 순서로 폴백했다고 말하는가."""
    c = Case("4. 리랭킹(CE) 실패")
    s.fault("rerank_fail")
    s.open()
    s.search()
    st = s.settled()
    s.shot("4_rerank_fail")
    c.check(len(st["results"]) > 0, f"결과는 그대로 뜬다 ({len(st['results'])}곡)")
    c.see(f"실행 기록: {st['runNote'] or '(없음)'}")
    c.check("실패" in st["runNote"], "리랭킹 실패를 화면이 말한다")
    c.check(st["runNoteWarn"], "경고 표시가 붙는다")
    x = _explain(s)
    c.see(f"1위 곡 설명: {x['summary']}")
    c.check(
        "리랭킹이 실패" in x["summary"] or "검색 순서" in x["summary"],
        "곡별 설명도 폴백을 말한다",
    )
    s.fault("ok")
    s.search()
    st = s.settled()
    c.check(not st["runNoteWarn"], "리랭커 복구 후 경고가 사라진다")
    return c


def case_empty(s: Screen) -> Case:
    """결과 0건 — 다음 행동이 분명한가, 막힌 버튼은 없는가."""
    c = Case("5. 결과 없음")
    s.fault("empty")
    s.open()
    s.search()
    st = s.settled()
    s.shot("5_empty")
    c.see(f"상태줄: {st['status'] or '(없음)'}")
    c.see(f"재질문 영역: {st['clarify'] or '(없음)'}")
    c.check(st["status"] != "" or st["clarify"] != "", "다음 행동을 안내한다")
    c.check(
        all(not b["disabled"] for b in st["buttons"]),
        f"막힌 버튼이 없다 (버튼 {len(st['buttons'])}개)",
    )
    c.check(st["panelMode"] != "loading", "로딩이 풀렸다")
    s.fault("ok")
    s.search()
    st = s.settled()
    c.check(len(st["results"]) > 0, "다시 검색하면 정상 동작한다")
    return c


def case_reject_exhausted(s: Screen) -> Case:
    """재질문·거절 한도 소진 — 끝났다는 것을 말하고 멈추는가."""
    c = Case("6. 재질문/거절 소진")
    s.fault("ok")
    s.open()
    s.search()
    s.settled()
    rounds = 0
    while rounds < 6:
        if not s.click("이 중에는 없어요"):
            break
        rounds += 1
        st = s.wait(lambda x: x["panelMode"] != "loading", timeout=20)
        if st["panelMode"] == "asking":
            # 질문이 떴으면 답해서 다음 턴으로 넘어간다
            answered = s.click("잘 모르겠어요")
            if answered:
                s.wait(lambda x: x["panelMode"] != "loading", timeout=20)
    st = s.snap()
    s.shot("6_exhausted")
    c.see(f"거절 {rounds}회 뒤 재질문 영역: {st['clarify'] or '(없음)'}")
    c.see(f"버튼: {[b['text'] for b in st['buttons']] or '(없음)'}")
    c.check(rounds > 0, f"거절이 동작한다 ({rounds}회)")
    c.check(st["clarify"] != "", "끝났다는 것을 문구로 말한다")
    c.check(
        all(not b["disabled"] for b in st["buttons"]),
        "눌리지 않는 버튼이 남지 않는다",
    )
    c.check(len(st["results"]) > 0, "마지막 결과는 화면에 남아 있다")
    return c


def case_out_of_order(s: Screen) -> Case:
    """연속 검색 — 늦게 온 옛 응답이 새 결과를 덮는가."""
    c = Case("7. 연속 검색 · 응답 순서 뒤바뀜")
    q_slow, q_fast = "느린 첫 질의", "빠른 두 번째 질의"

    # 두 질의가 서로 다른 곡을 내야 이 확인이 뜻을 가진다. 먼저 확인해 둔다.
    s.fault("ok")
    s.open()
    expect = {}
    for q in (q_slow, q_fast):
        s.search(q)
        expect[q] = list(s.settled()["results"])
    c.check(
        expect[q_slow] != expect[q_fast],
        "두 질의의 결과가 서로 다르다(이 확인이 성립하는 조건)",
    )

    s.fault("slow_first", seconds=6)
    s.open()
    s.search(q_slow)
    time.sleep(0.8)
    s.search(q_fast)                       # 두 번째는 즉시 돌아온다
    st = s.wait(lambda x: len(x["results"]) > 0, timeout=10)
    shown = list(st["results"])
    c.check(shown == expect[q_fast], "두 번째 질의의 결과가 화면에 떴다")
    time.sleep(7)                          # 첫 응답이 뒤늦게 도착할 시간
    st = s.snap()
    s.shot("7_out_of_order")
    c.check(
        st["results"] == shown and st["results"] != expect[q_slow],
        "늦게 도착한 첫 응답이 새 결과를 덮지 않는다",
    )
    c.check(st["status"] == "" or "검색 중" not in st["status"],
            f"상태줄이 남지 않는다 ({st['status'] or '비어 있음'})")
    return c


def case_server_error(s: Screen) -> Case:
    """500 — 안내하고, 다시 시도할 수 있는가."""
    c = Case("8. 서버 오류(500)")
    s.fault("server_error")
    s.open()
    s.search()
    st = s.settled()
    s.shot("8_server_error")
    c.see(f"상태줄: {st['status'] or '(없음)'}")
    c.check(st["status"] != "", "오류를 안내한다")
    c.check(
        st["status"] != EMPTY_MESSAGE,
        f"결과 0건과 구분되는 문구다 (0건일 때와 같은 '{EMPTY_MESSAGE}'가 아니다)",
    )
    c.check(st["panelMode"] != "loading", "로딩이 풀렸다")
    c.check(
        all(not b["disabled"] for b in st["buttons"]),
        "막힌 버튼이 없다",
    )
    s.fault("ok")
    s.search()
    st = s.settled()
    c.check(len(st["results"]) > 0, "복구 후 재검색이 정상 동작한다")
    return c


def case_rerank_fail_midturn(s: Screen) -> Case:
    """재질문 도중 실패 — 직전 화면으로 돌아오고 다시 누를 수 있는가."""
    c = Case("9. 재질문 도중 서버 오류")
    s.fault("ok")
    s.open()
    s.search()
    s.settled()
    before = s.snap()
    # "이 중에는 없어요"만으로는 요청이 나가지 않는다 — 질문은 이미 응답에 실려
    # 있어서 화면만 바뀐다. 답을 골라야 그때 다시 검색한다.
    c.check(s.click("이 중에는 없어요"), "거절 버튼을 누를 수 있다")
    st = s.wait(lambda x: x["panelMode"] == "asking", timeout=5)
    c.check(st["panelMode"] == "asking", "질문이 뜬다(여기까지는 요청 없음)")

    s.fault("server_error")
    c.check(s.click("잘 모르겠어요"), "답을 고를 수 있다")
    st = s.wait(lambda x: x["panelMode"] != "loading", timeout=20)
    s.shot("9_midturn_error")
    c.see(f"상태줄: {st['status'] or '(없음)'}")
    c.see(f"돌아온 화면: panelMode={st['panelMode']} · 버튼 {[b['text'] for b in st['buttons']]}")
    c.check(st["status"] != "", "오류를 안내한다")
    c.check(st["results"] == before["results"], "직전 결과가 그대로 남는다")
    c.check(
        all(not b["disabled"] for b in st["buttons"]),
        "버튼이 다시 눌리는 상태로 돌아온다",
    )
    s.fault("ok")
    c.check(s.click("잘 모르겠어요"), "같은 자리에서 재시도할 수 있다")
    st = s.wait(lambda x: x["panelMode"] != "loading", timeout=20)
    c.check(len(st["results"]) > 0, "재시도가 정상 동작한다")
    c.check(st["status"] == "", "복구되면 경고가 사라진다")
    return c


def case_warmup(s: Screen) -> Case:
    """예열 중 · 일부 예열 실패 — 검색과 안내가 정상인가."""
    c = Case("10. 예열 중 · 일부 예열 실패")
    s.fault("ok", warmup="running")
    s.open()
    health = json.loads(urllib.request.urlopen(f"{s.base}/health", timeout=5).read())
    c.see(f"/health warmup.state = {health['warmup']['state']}")
    s.search()
    st = s.settled()
    c.check(len(st["results"]) > 0, "예열 중에도 검색이 된다")
    s.fault("ok", warmup="partial_failure")
    health = json.loads(urllib.request.urlopen(f"{s.base}/health", timeout=5).read())
    failed = [k for k, v in health["warmup"]["stages"].items()
              if str(v.get("outcome", "")).startswith("failed")]
    c.see(f"/health가 실패한 단계를 알려준다: {failed}")
    c.check(bool(failed), "어느 단계가 실패했는지 /health로 알 수 있다")
    s.search()
    st = s.settled()
    s.shot("10_warmup")
    c.check(len(st["results"]) > 0, "일부 예열 실패 상태에서도 검색이 된다")
    c.check(
        "예열" not in st["status"],
        "화면은 예열 상태를 말하지 않는다(발표자는 /health로 본다)",
    )
    return c


def _explain(s: Screen) -> Dict[str, Any]:
    """1위 곡의 근거를 펼쳐서 읽는다. **화면이 그린 것**과 응답 원본을 같이 돌려준다."""
    s.d.execute_script(
        "const b = document.querySelector('#resultList .exbtn');"
        "if (b && b.getAttribute('aria-expanded') !== 'true') b.click();"
    )
    time.sleep(0.4)
    return s.d.execute_script(r"""
      const panel = document.querySelector("#resultList .explainrow .explain");
      const groups = panel ? [...panel.querySelectorAll(".exgroup")] : [];
      const rows = (title) => {
        const g = groups.find((x) => ((x.querySelector(".exhead") || {}).textContent || "")
                                      .startsWith(title));
        return g ? [...g.querySelectorAll(".exrow")].map(
          (r) => (r.textContent || "").replace(/\s+/g, " ").trim()) : [];
      };
      const last = window.__vfLast || {};
      const top = (last.results || [])[0] || {};
      return {
        summary: panel ? ((panel.querySelector(".exsummary") || {}).textContent || "").trim() : "",
        paths: rows("경로 기여"),
        groups: groups.map((g) => ((g.querySelector(".exhead") || {}).textContent || "").trim()),
        weights: (last.explain || {}).modality_weights || {},
        serverPaths: ((top.explain || {}).paths || []).map((p) => p.label),
        serverStage: (last.explain || {}).reorder_stage_label || "",
        confidence: (last.analysis || {}).confidence,
      };
    """)


CASES: List[Callable[[Screen], Case]] = [
    case_quota,
    case_hang,
    case_path_fail,
    case_rerank_fail,
    case_empty,
    case_reject_exhausted,
    case_out_of_order,
    case_server_error,
    case_rerank_fail_midturn,
    case_warmup,
]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def _wait_for(url: str, seconds: float = 30) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/__fault", timeout=2).read()
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="이미 떠 있는 모의 서버 주소")
    ap.add_argument("--start", action="store_true", help="모의 서버를 직접 띄운다")
    ap.add_argument("--headed", action="store_true", help="브라우저 창을 띄운다")
    ap.add_argument("--shots", default="", help="스크린샷을 남길 폴더")
    ap.add_argument("--only", default="", help="이름에 이 말이 든 항목만")
    args = ap.parse_args(argv)

    server = None
    if args.start or not _wait_for(args.url, seconds=1):
        server = subprocess.Popen(
            [sys.executable, "-m", "src.backend.demo_faults"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_for(args.url):
            print("모의 서버가 뜨지 않았다", file=sys.stderr)
            server.kill()
            return 2

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    if not args.headed:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,960")
    driver = webdriver.Chrome(options=options)
    screen = Screen(driver, args.url, Path(args.shots) if args.shots else None)

    cases: List[Case] = []
    try:
        for factory in CASES:
            if args.only and args.only not in (factory.__doc__ or "") \
                    and args.only not in factory.__name__:
                continue
            case = factory(screen)
            cases.append(case)
            print(f"\n{case.name}")
            for line in case.observations + case.failures:
                print("  " + line)
    finally:
        driver.quit()
        if server is not None:
            server.terminate()

    failed = [c for c in cases if c.failures]
    print("\n" + "=" * 70)
    print(f"확인 {len(cases)}항목 · 문제 {sum(len(c.failures) for c in failed)}건")
    for c in failed:
        print(f"  - {c.name}")
        for line in c.failures:
            print("      " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
