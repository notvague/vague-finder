/* 운영자용 피드백 검토 화면(로컬 전용).
 *
 * 원질의·사유는 사용자가 쓴 글이다. 모든 값은 textContent로만 넣는다 — innerHTML을 쓰지 않는다.
 * 저장은 화면이 본 제출 판(revision)·검토 판(review_version)을 함께 보내고, 서버가 그 사이 바뀐 것을
 * 알려 주면(409) 최신 내용을 다시 보여주되 작성 중이던 문장은 지우지 않는다.
 */
"use strict";

const STATE_LABELS = { pending: "대기", approved: "승인", on_hold: "보류", rejected: "제외", withdrawn: "철회", all: "전체" };
const FILTERS = ["pending", "approved", "on_hold", "rejected", "withdrawn", "all"];
const INDEX_LABELS = { pending: "색인 대기", indexed: "색인 완료", failed: "색인 실패(재시도 예정)" };
const TTL_LABELS = {
  ready: "준비됨", missing: "색인 없음", not_ttl: "TTL이 아닌 색인", wrong_seconds: "만료 초가 0이 아님",
  unauthorized: "권한 없음", unreachable: "확인 실패", unknown: "미확인",
};
const SLOT_LABELS = { vocal_gender: "가수 성별", genre: "장르", type: "가수 형태", artist_type: "가수 형태", release_era: "발매 시기" };
const PAGE_SIZE = 20;
// 이보다 짧은 재사용 단서에는 경고를 보인다(저장은 막지 않는다). 짧은 단서가 모두 넓은 것은 아니지만,
// "슬픈 발라드" 같은 넓은 단서는 다른 질의에도 쉽게 걸리고 유사도 문턱으로는 걸러지지 않는다.
const SHORT_TEXT_CHARS = 10;
const SHORT_TEXT_WARNING = "단서가 짧습니다. 장르·분위기만 적은 넓은 단서는 무관한 검색에도 걸립니다 — 이 곡을 가려 주는 단서가 아니면 보류하거나 제외하세요.";

function shortTextWarning(value) {
  const length = value.replace(/\s+/g, "").length;
  return length > 0 && length < SHORT_TEXT_CHARS ? SHORT_TEXT_WARNING : "";
}

function remainingText(seconds) {
  if (!(seconds > 0)) return "만료됨";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  if (days > 0) return `${days}일 ${hours}시간 남음`;
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours > 0 ? `${hours}시간 ${minutes}분 남음` : `${Math.max(1, minutes)}분 남음`;
}

function timeText(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "-";
  const pad = n => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function songText(song) {
  if (!song) return "선택 없음";
  const name = [song.title || `곡 ID ${song.id}`, song.artist].filter(Boolean).join(" — ");
  return song.rank ? `${name} (표시 ${song.rank}위)` : name;
}

class ReviewController {
  constructor(root, { api = ReviewController.api, now = () => Date.now() } = {}) {
    this.root = root;
    this.doc = root.ownerDocument;
    this.api = api;
    this.now = now;
    this.session = null;
    this.status = null;
    this.filter = "pending";
    this.page = 1;
    this.list = { items: [], total: 0 };
    this.listEpoch = 0;
    this.detailEpoch = 0;
    this.selectedId = null;
    this.detail = null;
    this.loadedAt = 0;          // 상세를 받은 시각. 남은 보관 기간을 다시 계산할 때 쓴다
    this.draft = { text: "", note: "" };
    this.message = "";
    this.messageKind = "";
    this.listError = "";
    this.saving = new Set();    // 저장 중인 사례 ID. 다른 사례를 열어도 그 사례의 버튼은 막지 않는다
    this.render();
  }

  // --- 서버 호출 -----------------------------------------------------------
  static async request(path, options = {}) {
    const res = await fetch(path, { cache: "no-store", credentials: "same-origin", ...options });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error?.message || "요청을 처리하지 못했습니다.");
      err.status = res.status;
      err.code = data.error?.code;
      err.current = data.current || null;
      throw err;
    }
    return data;
  }

  // --- 흐름 ---------------------------------------------------------------
  async start() {
    try {
      this.session = await this.api.session();
    } catch (err) {
      this.listError = err.message || "검토 화면을 시작하지 못했습니다.";
      this.render();
      return;
    }
    this.refreshStatus();
    await this.loadList();
  }

  async refreshStatus() {
    try {
      this.status = await this.api.status();
    } catch (err) {
      this.status = { error: "status_unavailable" };
    }
    this.renderHeader();
  }

  async loadList() {
    const epoch = ++this.listEpoch;
    this.listError = "";
    try {
      const data = await this.api.list(this.filter, this.page, PAGE_SIZE);
      if (epoch !== this.listEpoch) return;       // 필터·쪽을 바꾼 뒤 도착한 옛 응답
      this.list = { items: data.items || [], total: data.total || 0 };
    } catch (err) {
      if (epoch !== this.listEpoch) return;
      this.list = { items: [], total: 0 };
      this.listError = err.message || "목록을 불러오지 못했습니다.";
    }
    this.renderList();
  }

  setFilter(filter) {
    if (!FILTERS.includes(filter) || filter === this.filter) return;
    this.filter = filter;
    this.page = 1;
    this.renderList();
    return this.loadList();
  }

  setPage(page) {
    const last = Math.max(1, Math.ceil(this.list.total / PAGE_SIZE));
    const next = Math.min(last, Math.max(1, page));
    if (next === this.page) return;
    this.page = next;
    return this.loadList();
  }

  async open(searchId) {
    const epoch = ++this.detailEpoch;
    this.selectedId = searchId;
    this.detail = null;
    this.message = "불러오는 중…";
    this.messageKind = "";
    this.renderList();
    this.renderDetail();
    try {
      const detail = await this.api.detail(searchId);
      if (epoch !== this.detailEpoch) return;     // 다른 사례를 연 뒤 도착한 옛 응답
      this.show(detail, { resetDraft: true });
      this.message = "";
    } catch (err) {
      if (epoch !== this.detailEpoch) return;
      this.detail = null;
      this.message = err.message || "상세를 불러오지 못했습니다.";
      this.messageKind = "error";
    }
    this.renderDetail();
  }

  show(detail, { resetDraft }) {
    this.detail = detail;
    this.loadedAt = this.now();
    if (resetDraft) this.draft = { text: detail.reusable_text || "", note: detail.review?.note || "" };
  }

  remainingSeconds() {
    if (!this.detail) return 0;
    return (this.detail.remaining_seconds || 0) - Math.floor((this.now() - this.loadedAt) / 1000);
  }

  canReview() {
    return !!this.detail && this.detail.reviewable && this.detail.user_status === "pending" && this.remainingSeconds() > 0;
  }

  async save(state) {
    if (!this.detail || this.saving.has(this.detail.search_id)) return;
    if (!this.canReview()) {
      this.message = this.blockedReason();
      this.messageKind = "error";
      this.renderDetail();
      return;
    }
    const text = this.draft.text.replace(/\s+/g, " ").trim();
    if (state === "approved" && !text) {
      this.message = "승인하려면 검색에 재사용할 단서를 적어야 합니다. 정리할 단서가 없으면 보류하거나 제외하세요.";
      this.messageKind = "error";
      this.renderDetail();
      return;
    }
    const epoch = this.detailEpoch;
    const searchId = this.detail.search_id;
    this.saving.add(searchId);
    this.message = "저장 중…";
    this.messageKind = "";
    this.renderDetail();
    let saved = null;
    let err = null;
    try {
      saved = await this.api.review(searchId, {
        state, reusable_text: this.draft.text, note: this.draft.note,
        expected_revision: this.detail.revision, expected_review_version: this.detail.review_version,
      }, this.session?.csrf_token);
    } catch (failure) {
      err = failure;
    }
    this.saving.delete(searchId);
    this.loadList();
    this.refreshStatus();
    if (epoch !== this.detailEpoch) return;       // 저장하는 동안 다른 사례를 열었다. 그 화면은 건드리지 않는다
    if (saved) {
      this.show(saved, { resetDraft: true });
      this.message = `${STATE_LABELS[saved.review_state] || saved.review_state}(으)로 저장했습니다.`;
      this.messageKind = "ok";
    } else {
      if (err.current) {
        // 그 사이 사용자나 다른 검토가 문서를 바꿨다. 최신 내용을 보여주되 **작성 중이던 입력은 그대로 둔다.**
        this.show(err.current, { resetDraft: false });
        this.message = `${err.message} 작성 중이던 내용은 그대로 두었습니다. 최신 내용을 확인한 뒤 다시 저장하세요.`;
      } else if (err.status === 404 || err.status === 410) {
        this.detail = { ...this.detail, reviewable: false, remaining_seconds: 0 };
        this.message = err.message;
      } else {
        this.message = `${err.message || "저장하지 못했습니다."} 입력은 그대로 있습니다. 다시 시도해 주세요.`;
      }
      this.messageKind = "error";
    }
    this.renderDetail();
  }

  blockedReason() {
    if (!this.detail) return "";
    if (this.detail.user_status !== "pending") return "사용자가 철회한 사례는 검토할 수 없습니다.";
    return "보관 기간이 지난 사례는 검토할 수 없습니다.";
  }

  // --- 그리기 ---------------------------------------------------------------
  el(tag, { cls, text, attrs, on } = {}, ...children) {
    const node = this.doc.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    for (const [key, handler] of Object.entries(on || {})) node.addEventListener(key, handler);
    node.append(...children.filter(Boolean));
    return node;
  }

  /** 자식을 갈아 끼운다. 없는 자리(null)는 뺀다 — 실제 DOM은 null을 "null"이라는 글자로 넣는다. */
  fill(node, ...children) {
    node.replaceChildren(...children.filter(Boolean));
  }

  render() {
    this.header = this.el("header", { cls: "review-header" });
    this.listPane = this.el("section", { cls: "review-list", attrs: { "aria-label": "검토 목록" } });
    this.detailPane = this.el("section", { cls: "review-detail", attrs: { "aria-label": "검토 상세" } });
    this.fill(this.root, this.header, this.el("div", { cls: "review-body" }, this.listPane, this.detailPane));
    this.renderHeader();
    this.renderList();
    this.renderDetail();
  }

  renderHeader() {
    const reviewer = this.session
      ? `검토자 표기: ${this.session.reviewer} (로컬 설정값 · 확인된 신원이 아닙니다)`
      : "";
    this.fill(this.header, 
      this.el("h1", { text: "피드백 검토" }),
      this.el("p", { cls: "review-scope", text: "로컬 전용 · 인증 없음 · 자동 승인 없음" }),
      this.el("p", { cls: "review-reviewer", text: reviewer }),
      this.statusLine(),
    );
  }

  statusLine() {
    const status = this.status;
    if (!status) return this.el("p", { cls: "review-status", text: "운영 상태 확인 중…" });
    if (status.error && !status.ttl) return this.el("p", { cls: "review-status warn", text: "운영 상태를 확인하지 못했습니다." });
    const parts = [];
    const ttl = Object.entries(status.ttl || {}).map(([name, value]) => `${name} ${TTL_LABELS[value] || value}`).join(", ");
    parts.push(status.ttl_ready ? "자동 삭제(TTL) 준비됨" : `자동 삭제(TTL) 준비 안 됨 — ${ttl}`);
    if (status.index) parts.push(`승인 ${status.index.approved} · 색인 완료 ${status.index.indexed} · 색인 실패 ${status.index.failed}`);
    const worker = status.worker;
    parts.push(worker?.last_success_at ? `색인 작업 마지막 성공 ${timeText(worker.last_success_at)}` : "색인 작업 기록 없음(검색 서버에서 켜야 색인됩니다)");
    if (worker?.last_error) parts.push(`최근 오류: ${worker.last_error}`);
    const expired = status.expired_remaining;
    if (expired && (expired.search_runs || expired.search_feedback)) {
      parts.push(`만료 후 남은 문서 ${expired.search_runs + expired.search_feedback}건`);
    }
    const warn = !status.ttl_ready || !!worker?.last_error;
    return this.el("p", { cls: warn ? "review-status warn" : "review-status", text: parts.join(" · ") });
  }

  renderList() {
    const tabs = this.el("div", { cls: "review-tabs", attrs: { role: "tablist" } }, ...FILTERS.map(filter =>
      this.el("button", {
        cls: filter === this.filter ? "tab active" : "tab", text: STATE_LABELS[filter],
        attrs: { type: "button", role: "tab", "aria-selected": String(filter === this.filter), "data-filter": filter },
        on: { click: () => this.setFilter(filter) },
      })));
    const rows = this.list.items.map(item => this.listRow(item));
    const last = Math.max(1, Math.ceil(this.list.total / PAGE_SIZE));
    const pager = this.el("div", { cls: "review-pager" },
      this.pageButton("이전", this.page - 1, this.page <= 1),
      this.el("span", { text: `${this.page} / ${last}쪽 · 전체 ${this.list.total}건` }),
      this.pageButton("다음", this.page + 1, this.page >= last));
    const empty = this.listError
      ? this.el("p", { cls: "review-message error", text: this.listError })
      : (rows.length ? null : this.el("p", { cls: "review-empty", text: "이 상태의 사례가 없습니다." }));
    this.fill(this.listPane, tabs, empty, this.el("ul", { cls: "review-rows" }, ...rows), pager);
  }

  pageButton(label, page, disabled) {
    const button = this.el("button", { text: label, attrs: { type: "button" }, on: { click: () => this.setPage(page) } });
    button.disabled = disabled;
    return button;
  }

  listRow(item) {
    const state = item.user_status === "withdrawn" ? "withdrawn" : item.review_state;
    const meta = [
      item.has_reason ? "사유 있음" : "사유 없음",
      `생성 ${timeText(item.created_at)}`,
      remainingText(item.remaining_seconds),
      INDEX_LABELS[item.index_state] || "",
    ].filter(Boolean).join(" · ");
    return this.el("li", {},
      this.el("button", {
        cls: item.search_id === this.selectedId ? "row active" : "row",
        attrs: { type: "button", "data-id": item.search_id },
        on: { click: () => this.open(item.search_id) },
      },
        this.el("span", { cls: `chip ${state}`, text: STATE_LABELS[state] || state }),
        this.el("span", { cls: "row-song", text: songText(item.selected_song) }),
        this.el("span", { cls: "row-meta", text: meta })));
  }

  field(label, ...children) {
    return this.el("div", { cls: "field" }, this.el("h3", { text: label }), ...children);
  }

  renderDetail() {
    const detail = this.detail;
    const message = this.message
      ? this.el("p", { cls: `review-message ${this.messageKind}`.trim(), text: this.message, attrs: { role: "status" } })
      : null;
    if (!detail) {
      this.fill(this.detailPane, message || this.el("p", { cls: "review-empty", text: "목록에서 사례를 선택하세요." }));
      return;
    }
    const context = detail.context || {};
    const answers = (context.answers || []).map(answer =>
      `${SLOT_LABELS[answer.slot] || answer.slot}: ${answer.skipped ? "잘 모르겠어요" : answer.value}`);
    const rejected = (context.rejected || []).map(song => [song.title || `곡 ID ${song.id}`, song.artist].filter(Boolean).join(" — "));
    const era = context.release_era && (context.release_era.start_year || context.release_era.end_year)
      ? `${context.release_era.start_year || "?"}~${context.release_era.end_year || "?"}년` : "";
    const contextLines = [
      context.turn ? `${context.turn}번째 턴의 검색` : "",
      answers.length ? `재질문 답변 — ${answers.join(", ")}` : "재질문 답변 없음",
      rejected.length ? `거절한 곡 ${rejected.length}개 — ${rejected.join(" / ")}` : "거절한 곡 없음",
      era ? `검색에 쓴 발매 시기 창 — ${era}` : "",
    ].filter(Boolean);

    const current = detail.current_song;
    let currentText = "선택한 곡이 없습니다.";
    if (current) {
      if (current.found === true) {
        currentText = [current.title, current.artist, current.album, current.release_date, current.genre].filter(Boolean).join(" · ");
      } else if (current.found === false) {
        currentText = "현재 곡 목록에서 찾지 못했습니다. 삭제된 곡이면 승인해도 검색에 쓰이지 않습니다.";
      } else {
        currentText = "현재 곡 정보를 읽지 못했습니다. 검색 당시 정보만 확인할 수 있습니다.";
      }
    }
    const review = detail.review;
    const reviewText = review
      ? `${STATE_LABELS[review.state] || review.state} · 검토 판 ${review.version} · ${review.reviewed_by || "?"}(로컬 설정값) · ${timeText(review.reviewed_at)}`
      : "아직 검토하지 않았습니다.";
    const history = (detail.review_history || []).map(event =>
      `${timeText(event.reviewed_at)} ${STATE_LABELS[event.state] || event.state} (제출 판 ${event.source_revision} · 검토 판 ${event.version})`);

    this.fill(this.detailPane, 
      this.el("h2", { text: songText(detail.selected_song) }),
      this.el("p", { cls: "detail-meta", text:
        `${STATE_LABELS[detail.user_status === "withdrawn" ? "withdrawn" : detail.review_state]} · 제출 판 ${detail.revision} · 검토 판 ${detail.review_version}` }),
      this.field("원질의", this.el("p", { cls: "quote", text: detail.run_found ? (detail.query || "") : "검색 기록을 찾지 못했습니다." })),
      this.field("재질문 맥락", this.el("ul", {}, ...contextLines.map(line => this.el("li", { text: line })))),
      this.field("선택한 곡",
        this.el("p", { text: `검색 당시: ${songText(detail.selected_song)}` }),
        this.el("p", { cls: current && current.found === true ? "" : "warn", text: `현재 곡 정보: ${currentText}` })),
      this.field("사용자 사유", this.el("p", { cls: "quote", text: detail.reason || "사유를 적지 않았습니다." })),
      this.field("보관 기간",
        this.el("p", { text: `생성 ${timeText(detail.created_at)} · 만료 ${timeText(detail.expires_at)} · ${remainingText(this.remainingSeconds())}` }),
        this.el("p", { cls: "hint", text: "검색일부터 30일 뒤 원문·사유·검토 결과·재사용 문장이 함께 삭제됩니다. 검토·승인으로 기간이 늘지 않습니다." })),
      this.field("검토 결과",
        this.el("p", { text: reviewText }),
        history.length ? this.el("ul", { cls: "history" }, ...history.map(line => this.el("li", { text: line }))) : null),
      this.editor(),
      message,
    );
  }

  editor() {
    const limits = this.session?.limits || { reusable_text: 500, note: 1000 };
    const enabled = this.canReview() && !this.saving.has(this.detail.search_id);
    const counter = (value, max) => `${value.length} / ${max}자`;

    const textCount = this.el("span", { cls: "count", text: counter(this.draft.text, limits.reusable_text) });
    const text = this.el("textarea", { attrs: { rows: "4", maxlength: String(limits.reusable_text), "data-role": "reusable-text", "aria-label": "검색에 재사용할 단서" } });
    text.value = this.draft.text;
    const textWarning = this.el("p", { cls: "hint warn", text: shortTextWarning(this.draft.text), attrs: { "data-role": "text-warning" } });
    text.addEventListener("input", () => {
      this.draft.text = text.value;
      textCount.textContent = counter(text.value, limits.reusable_text);
      textWarning.textContent = shortTextWarning(text.value);
    });
    const noteCount = this.el("span", { cls: "count", text: counter(this.draft.note, limits.note) });
    const note = this.el("textarea", { attrs: { rows: "2", maxlength: String(limits.note), "data-role": "review-note", "aria-label": "검토 사유" } });
    note.value = this.draft.note;
    note.addEventListener("input", () => {
      this.draft.note = note.value;
      noteCount.textContent = counter(note.value, limits.note);
    });
    text.disabled = note.disabled = !enabled;

    const action = (state, label, cls) => {
      const button = this.el("button", { cls, text: label, attrs: { type: "button", "data-action": state }, on: { click: () => this.save(state) } });
      button.disabled = !enabled;
      return button;
    };
    const blocked = this.canReview() ? null : this.el("p", { cls: "review-message error", text: this.blockedReason() });
    return this.el("div", { cls: "editor" },
      this.field("검색에 재사용할 단서(승인하려면 필요)", text, textCount, textWarning,
        this.el("p", { cls: "hint", text:
          "질의와 사유에서 다음 검색에 다시 쓸 수 있는 단서만 직접 정리합니다. 개인 기억·개인정보·틀린 기억(성별·시기 등)은 넣지 않습니다. "
          + "‘슬픈 발라드’처럼 넓은 단서뿐이면 보류하거나 제외합니다. 선택은 이 사용자가 찾던 곡이라는 뜻이며 곡의 사실을 확인한 것은 아닙니다." })),
      this.field("검토 사유(선택)", note, noteCount),
      blocked,
      this.el("div", { cls: "actions" },
        action("approved", "승인", "primary"), action("on_hold", "보류"), action("rejected", "제외"),
        action("pending", "대기로 되돌리기", "quiet")));
  }
}

ReviewController.api = {
  session: () => ReviewController.request("/api/session"),
  status: () => ReviewController.request("/api/status"),
  list: (state, page, pageSize) => ReviewController.request(
    `/api/feedback?state=${encodeURIComponent(state)}&page=${page}&page_size=${pageSize}`),
  detail: id => ReviewController.request(`/api/feedback/${encodeURIComponent(id)}`),
  review: (id, body, token) => ReviewController.request(`/api/feedback/${encodeURIComponent(id)}/review`, {
    method: "PUT", headers: { "Content-Type": "application/json", "X-Review-Token": token || "" },
    body: JSON.stringify(body),
  }),
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = { ReviewController, remainingText, songText, shortTextWarning };
} else if (typeof document !== "undefined") {
  const root = document.getElementById("review-root");
  if (root) new ReviewController(root).start();
}
