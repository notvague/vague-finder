/* 정답 피드백. 제출 권한·원문은 이 화면의 메모리에만 둔다. 검색 순위는 바꾸지 않는다. */
"use strict";

class FeedbackController {
  constructor(root, { send = FeedbackController.send, key = FeedbackController.key, onChange = () => {} } = {}) {
    this.root = root;
    this.send = send;
    this.key = key;
    this.onChange = onChange;
    this.epoch = 0;
    this.buttons = [];
    this.reset(null);
  }

  static key() {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, n => n.toString(16).padStart(2, "0")).join("");
  }

  static async send(body) {
    const res = await fetch("/api/v1/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: AbortSignal.timeout(5000), cache: "no-store",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(typeof data.detail === "string" ? data.detail : data.detail?.message || "저장 여부를 확인하지 못했습니다.");
      err.status = res.status;
      err.current = data.detail?.current;
      throw err;
    }
    return data;
  }

  reset(data) {
    this.epoch += 1;
    this.receipt = data?.feedback || null;
    this.tracks = data?.results || [];
    this.saved = null;
    this.draft = null;
    this.pending = null;
    this.sending = false;
    this.busy = false;
    this.message = "";
    this.buttons = [];
    this.render();
  }

  setBusy(busy) {
    this.busy = busy;
    // 결과 행은 renderPanel이 새로 만들므로 옛 버튼 참조를 버린다.
    this.buttons = [];
    this.render();
  }

  attach(container, track) {
    if (!this.receipt) return;
    const button = this.element("button", "feedbackpick", "찾던 곡이에요");
    button.type = "button";
    button.setAttribute("aria-label", `${track.title} — 찾던 곡이에요`);
    button.disabled = this.busy || this.sending;
    button.addEventListener("click", event => {
      event.stopPropagation();
      this.choose(String(track.id));
    });
    this.buttons.push(button);
    container.append(button);
  }

  choose(id) {
    if (!this.receipt || this.busy || this.sending || !this.tracks.some(t => String(t.id) === id)) return;
    this.draft = { id, reason: this.saved?.selected_song_id === id ? this.saved.reason : "" };
    this.pending = null;
    this.message = "";
    this.render();
    this.root.querySelector("textarea")?.focus();
  }

  element(tag, className, text) {
    const el = this.root.ownerDocument.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  action(text, fn) {
    const button = this.element("button", "feedbackaction", text);
    button.type = "button";
    button.disabled = this.busy || this.sending;
    button.addEventListener("click", event => { event.stopPropagation(); fn(); });
    return button;
  }

  async submit(action = "select") {
    if (!this.receipt || this.busy || this.sending || (action === "select" && !this.draft)) return;
    const payload = {
      search_id: this.receipt.search_id, token: this.receipt.token,
      expected_revision: this.saved?.revision || 0, action,
      selected_song_id: action === "select" ? this.draft.id : null,
      reason: action === "select" ? this.draft.reason.trim() : "",
    };
    if (payload.reason.length > 1000) { this.message = "사유는 1,000자 안으로 적어 주세요."; this.render(); return; }
    // 응답이 유실됐어도 같은 내용의 재시도는 같은 제출 번호를 쓴다.
    const signature = JSON.stringify(payload);
    if (this.pending?.signature !== signature) this.pending = { signature, body: { ...payload, submission_id: this.key() } };
    const epoch = this.epoch;
    this.sending = true;
    this.message = "";
    this.render();
    try {
      const data = await this.send(this.pending.body);
      if (epoch !== this.epoch) return;
      this.saved = data;
      this.draft = null;
      this.pending = null;
    } catch (err) {
      if (epoch !== this.epoch) return;
      if (err.current) { this.saved = err.current; this.pending = null; }
      this.message = err.status === 409 ? "저장된 선택을 확인해 주세요. 지금 입력한 내용으로 바꾸려면 다시 제출해 주세요."
        : err.status === 404 ? "이 검색에는 더 이상 제출할 수 없습니다. 다시 검색해 주세요."
        : "저장 여부를 확인하지 못했습니다. 입력은 남아 있어요. 다시 제출해 주세요.";
    } finally {
      if (epoch === this.epoch) { this.sending = false; this.render(); }
    }
  }

  render() {
    this.root.replaceChildren();
    this.root.hidden = !this.receipt || (!this.draft && !this.saved && !this.message);
    for (const button of this.buttons) button.disabled = this.busy || this.sending;
    if (this.root.hidden) { this.onChange(); return; }
    if (this.saved) {
      const track = this.tracks.find(t => String(t.id) === this.saved.selected_song_id);
      const text = this.saved.status === "withdrawn" ? "정답 선택을 철회했어요."
        : `정답을 저장했어요: ${track?.title || this.saved.selected_song_id} · 검토 후 검색 개선에 활용합니다.`;
      const status = this.element("p", "feedbackstatus", text);
      status.setAttribute("role", "status");
      this.root.append(status);
      if (this.saved.status !== "withdrawn") this.root.append(this.action("선택 철회", () => this.submit("withdraw")));
    }
    if (this.draft) {
      const track = this.tracks.find(t => String(t.id) === this.draft.id);
      this.root.append(this.element("p", "feedbacktitle", `${track.title} · ${track.artist || ""}`));
      const label = this.element("label", "feedbacklabel", "어떤 단서로 알아보셨나요? (선택)");
      label.htmlFor = "feedbackReason";
      const input = this.element("textarea", "feedbackreason");
      input.id = "feedbackReason";
      input.maxLength = 1000;
      input.rows = 3;
      input.value = this.draft.reason;
      input.disabled = this.busy || this.sending;
      input.placeholder = "예: 도입부의 휘파람 소리로 알아봤어요.";
      input.addEventListener("input", () => { this.draft.reason = input.value; });
      input.addEventListener("keydown", event => {
        if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); if (!this.sending) { this.draft = null; this.render(); } }
      });
      this.root.append(label, input,
        this.element("p", "feedbacknote", `이 검색 기록과 제출한 선택·사유는 검색일부터 ${this.receipt.retention_days}일 보관해 검색 개선에 활용합니다. 개인정보는 적지 마세요.`));
      const actions = this.element("div", "feedbackactions");
      actions.append(this.action(this.sending ? "저장 중…" : "정답 제출", () => this.submit()),
        this.action("닫기", () => { this.draft = null; this.render(); }));
      this.root.append(actions);
    }
    if (this.message) {
      const note = this.element("p", "feedbackerror", this.message);
      note.setAttribute("role", "alert");
      this.root.append(note);
    }
    this.onChange();
  }
}

if (typeof module !== "undefined" && module.exports) module.exports = { FeedbackController };
else window.VagueFeedback = new FeedbackController(document.getElementById("feedbackBox"), {
  onChange: () => { if (typeof syncRailMetrics === "function") syncRailMetrics(); },
});
