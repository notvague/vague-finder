/*
 * 로딩 인트로 컨트롤러. 모양은 intro.css, 여기는 단계(off → run → exit → done)와 시간만 맡는다.
 *
 * 오버레이(#vagueIntro)는 index.html 에 처음부터 들어 있어 첫 페인트부터 화면을 덮는다.
 * 아이콘의 점·V는 아래 buildIcon()이 채운다.
 *
 * 앱 쪽 연결: 지도가 실제로 그려질 준비가 되면 VagueIntro.ready() 를 부른다(map.js init()).
 * exit 진입 시점 = max(최소 표시 시간, ready 시점). ready 가 영영 안 와도 maxWaitMs 뒤에는 나간다.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VagueIntro = api;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  // 조절하기 쉽게 파일 위쪽에 모은 기본값 (옵션으로도 덮어쓴다)
  const DEFAULTS = {
    glow: 1,                // 헤일로 세기 (0~1.5)
    dotStep: 110,           // 점 사이 점등 간격(ms)
    flyMs: 640,             // 비행 시간(ms)
    maxWaitMs: 10000,       // ready 가 안 올 때 강제로 나가는 시점
    oncePerSession: false,  // true 면 세션당 한 번만 보인다 (sessionStorage)
    storageKey: 'vagueIntroSeen',
  };

  const OFF_MS = 80;
  const DONE_MARGIN_MS = 60;
  const REDUCED_FADE_MS = 400;
  const MAX_WAIT_CAP_MS = 10000; // intro.css 의 vi-failsafe(12s)와 짝

  // 아이콘 박스 대비 %. i 는 점등 순서 (SPEC.md 4번)
  const DOT_ROWS = [
    { left: 6.287, right: 93.713, y: 6.184, w: 7.018, i: 0 },
    { left: 11.988, right: 87.865, y: 17.237, w: 8.48, i: 1 },
    { left: 18.275, right: 81.725, y: 29.079, w: 10.819, i: 2 },
    { left: 25.146, right: 74.854, y: 42.5, w: 13.158, i: 3 },
  ];

  const V_PATH = 'M396 566 L512 818 L628 566';

  /** 옵션에서 모든 타이밍을 한 번에 계산한다. */
  function computeTiming(options) {
    const o = Object.assign({}, DEFAULTS, options);
    const vDelay = 400 + 3 * o.dotStep + 170;
    const holdStart = vDelay + 560;
    return {
      glow: o.glow,
      dotStep: o.dotStep,
      flyMs: o.flyMs,
      vDelay,
      flashDelay: vDelay + 140,
      holdStart,
      minShow: holdStart + 400,
      sweepDelay: Math.round(o.flyMs * 0.3125),
      sweepMs: Math.round(o.flyMs * 1.09),
    };
  }

  function svgV(doc, lit) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = doc.createElementNS(ns, 'svg');
    svg.setAttribute('class', lit ? 'vi-v vi-vlit' : 'vi-v');
    svg.setAttribute('viewBox', '170 130 684 760');
    svg.setAttribute('aria-hidden', 'true');
    const path = doc.createElementNS(ns, 'path');
    path.setAttribute('d', V_PATH);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', lit ? '#ffffff' : '#232323');
    path.setAttribute('stroke-width', '108');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);
    return svg;
  }

  /** 아이콘 박스 안을 SPEC.md 3번의 순서대로 채운다. */
  function buildIcon(doc, icon, timing) {
    const div = (cls, style) => {
      const el = doc.createElement('div');
      el.className = cls;
      if (style) el.style.cssText = style;
      return el;
    };
    const haloStack = (wrapCls, haloCls, haloStyle) => {
      const wrap = div(wrapCls);
      const gain = div('vi-gain');
      gain.appendChild(div(haloCls, haloStyle));
      wrap.appendChild(gain);
      return wrap;
    };

    icon.replaceChildren();
    icon.appendChild(haloStack('vi-vhalo-wrap', 'vi-vhalo'));
    icon.appendChild(div('vi-vflash'));
    icon.appendChild(div('vi-vtrail', 'left:33.04%'));
    icon.appendChild(div('vi-vtrail', 'left:66.96%'));

    for (const row of DOT_ROWS) {
      for (const x of [row.left, row.right]) {
        const m = -(row.w / 2);
        const dot = div(
          'vi-dot',
          `left:${x}%;top:${row.y}%;width:${row.w}%;margin:${m}% 0 0 ${m}%;` +
            `--d:${400 + row.i * timing.dotStep}ms`
        );
        dot.appendChild(div('vi-trail'));
        dot.appendChild(haloStack('vi-halo-wrap', 'vi-halo'));
        dot.appendChild(div('vi-core'));
        dot.appendChild(div('vi-lit'));
        icon.appendChild(dot);
      }
    }
    icon.appendChild(svgV(doc, false));
    icon.appendChild(svgV(doc, true));
  }

  /**
   * @param {object} deps  { el, icon?, doc?, options?, setTimeout?, clearTimeout?, reducedMotion?,
   *                         storage?, onComplete? }
   *   테스트에서는 el·타이머를 가짜로 넣는다.
   */
  class Intro {
    constructor(deps) {
      this.el = deps.el;
      this.icon = deps.icon || null;
      this.doc = deps.doc || null;
      this.opts = Object.assign({}, DEFAULTS, deps.options);
      // intro.css 의 최후 안전장치(12s)보다 먼저 나가도록 상한을 둔다
      this.opts.maxWaitMs = Math.min(this.opts.maxWaitMs, MAX_WAIT_CAP_MS);
      this.timing = computeTiming(this.opts);
      this.setT = deps.setTimeout || setTimeout.bind(globalThis);
      this.clearT = deps.clearTimeout || clearTimeout.bind(globalThis);
      this.reduced = !!deps.reducedMotion;
      this.storage = deps.storage || null;
      this.onComplete = deps.onComplete || null;
      this.phase = 'idle';
      this._timers = [];
      this._ready = false;
      this._startedAt = 0;
      this._exitScheduled = false;
      this._now = deps.now || (() => Date.now());
    }

    _later(ms, fn) {
      this._timers.push(this.setT(fn, ms));
    }

    _setPhase(phase) {
      const cl = this.el.classList;
      for (const p of ['off', 'run', 'exit']) cl.remove('vi-' + p);
      if (phase === 'off' || phase === 'run' || phase === 'exit') cl.add('vi-' + phase);
      this.phase = phase;
    }

    /** 한 번만 시작한다(두 번 불려도 타이머가 겹치지 않는다). */
    start() {
      if (this.phase !== 'idle') return;

      // 안전장치를 가장 먼저 건다 — 아래 어느 단계에서 예외가 나도 검정에 갇히지 않게.
      // ready 가 끝내 안 오는 경우도 여기서 막는다. 먼저 phase 를 바꿔 중복 호출을 막는다.
      this.phase = 'starting';
      this._later(this.opts.maxWaitMs, () => this._exitOrFinish());

      if (this.opts.oncePerSession && this.storage) {
        let seen = false;
        try { seen = this.storage.getItem(this.opts.storageKey) === '1'; } catch (_) { /* 저장소 차단 */ }
        if (seen) { this._finish(); return; }
      }

      const t = this.timing;
      const style = this.el.style;
      style.setProperty('--vi-fly', t.flyMs + 'ms');
      style.setProperty('--vi-sweep-delay', t.sweepDelay + 'ms');
      style.setProperty('--vi-sweep-ms', t.sweepMs + 'ms');
      style.setProperty('--vi-v-delay', t.vDelay + 'ms');
      style.setProperty('--vi-halo-hold', t.holdStart + 'ms');
      style.setProperty('--vi-flash-delay', t.flashDelay + 'ms');
      style.setProperty('--vi-gain-opacity', String(Math.min(1, t.glow)));
      style.setProperty('--vi-gain-scale', String(0.7 + 0.3 * t.glow));
      try {
        if (this.icon && this.doc) buildIcon(this.doc, this.icon, t);
      } catch (_) {
        // 아이콘을 못 그리는 환경(replaceChildren 없음 등)이면 인트로를 건너뛴다
        this._finish();
        return;
      }

      this._setPhase('off');
      this._startedAt = this._now();
      this._later(OFF_MS, () => {
        this._setPhase('run');
        this._runAt = this._now();
        this._scheduleExit();
      });
    }

    /** 안전장치용: 아직 나가는 중이 아니면 나간다. 시작도 못 했으면 바로 걷어낸다. */
    _exitOrFinish() {
      if (this.phase === 'starting') this._finish();
      else this._exit();
    }

    /** 앱이 첫 화면을 그릴 준비가 됐다는 신호. */
    ready() {
      if (this._ready) return;
      this._ready = true;
      this._scheduleExit();
    }

    _scheduleExit() {
      if (this.phase !== 'run' || !this._ready || this._exitScheduled) return;
      this._exitScheduled = true;
      // run 진입 후 minShow 가 지난 뒤에 나간다. ready 가 더 늦었으면 바로.
      const wait = Math.max(0, this.timing.minShow - (this._now() - this._runAt));
      this._later(wait, () => this._exit());
    }

    _exit() {
      if (this.phase === 'exit' || this.phase === 'done' || this.phase === 'idle') return;
      this._setPhase('exit');
      this._exitScheduled = true;
      const t = this.timing;
      const total = this.reduced
        ? REDUCED_FADE_MS + DONE_MARGIN_MS
        : t.sweepDelay + t.sweepMs + DONE_MARGIN_MS;
      this._later(total, () => this._finish());
    }

    _finish() {
      if (this.phase === 'done') return;
      this.destroy();
      this.phase = 'done';
      if (this.opts.oncePerSession && this.storage) {
        try { this.storage.setItem(this.opts.storageKey, '1'); } catch (_) { /* 무시 */ }
      }
      if (this.onComplete) this.onComplete();
    }

    /** 타이머·오버레이를 모두 걷어낸다. */
    destroy() {
      for (const id of this._timers) this.clearT(id);
      this._timers = [];
      if (this.el.remove) this.el.remove();
    }
  }

  /** 페이지에서 쓰는 진입점: #vagueIntro 를 찾아 시작하고, 전역 ready() 를 노출한다. */
  function mount(win, options) {
    const doc = win.document;
    const el = doc.getElementById('vagueIntro');
    const unlock = () => doc.documentElement.classList.remove('vi-lock');
    if (!el) { unlock(); return null; }

    const reduced = !!(win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches);
    let storage = null;
    try { storage = win.sessionStorage; } catch (_) { /* 차단된 환경 */ }

    try {
      const intro = new Intro({
        el,
        icon: el.querySelector('.vi-icon'),
        doc,
        options,
        reducedMotion: reduced,
        storage,
        onComplete() {
          unlock();
          win.dispatchEvent(new win.CustomEvent('vagueintro:complete'));
        },
      });
      intro.start();
      return intro;
    } catch (_) {
      // 인트로는 장식이다. 어떤 이유로든 못 돌리면 앱을 가리지 않고 물러난다.
      if (el.remove) el.remove();
      unlock();
      return null;
    }
  }

  return { Intro, computeTiming, mount, DOT_ROWS, DEFAULTS };
});

// 브라우저에서는 로드되자마자 시작한다 (index.html 의 data-* 로 옵션을 덮어쓸 수 있다)
if (typeof window !== 'undefined' && window.document && !(typeof module !== 'undefined' && module.exports)) {
  (function () {
    const el = document.getElementById('vagueIntro');
    const ds = (el && el.dataset) || {};
    const num = (v) => (v === undefined || v === '' || isNaN(Number(v)) ? undefined : Number(v));
    const options = {};
    for (const k of ['glow', 'dotStep', 'flyMs', 'maxWaitMs']) {
      const v = num(ds[k]);
      if (v !== undefined) options[k] = v;
    }
    if (ds.oncePerSession === 'true') options.oncePerSession = true;

    const intro = window.VagueIntro.mount(window, options);
    const api = window.VagueIntro;
    api.instance = intro;
    api.ready = function () { if (intro) intro.ready(); };
  })();
}
