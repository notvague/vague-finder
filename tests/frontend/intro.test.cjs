const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Intro, computeTiming } = require('../../src/frontend/static/intro.js');

function harness(options = {}, extra = {}) {
  let now = 0;
  let nextId = 1;
  const pending = new Map();
  const classes = new Set(['vi-off']);
  const el = {
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
    },
    style: { props: {}, setProperty(k, v) { this.props[k] = v; } },
    removed: false,
    remove() { this.removed = true; },
  };
  const calls = { complete: 0 };
  const intro = new Intro({
    el,
    options,
    now: () => now,
    setTimeout: (fn, ms) => { const id = nextId++; pending.set(id, { at: now + ms, fn }); return id; },
    clearTimeout: (id) => pending.delete(id),
    onComplete: () => { calls.complete++; },
    ...extra,
  });
  const advance = (ms) => {
    const target = now + ms;
    for (;;) {
      let nextId_ = null;
      for (const [id, t] of pending) if (t.at <= target && (nextId_ === null || t.at < pending.get(nextId_).at)) nextId_ = id;
      if (nextId_ === null) break;
      const t = pending.get(nextId_);
      pending.delete(nextId_);
      now = t.at;
      t.fn();
    }
    now = target;
  };
  return { intro, el, classes, calls, advance, pending };
}

test('기본 타이밍은 SPEC 값과 같다', () => {
  const t = computeTiming();
  assert.equal(t.vDelay, 900);
  assert.equal(t.holdStart, 1460);
  assert.equal(t.minShow, 1860);
  assert.equal(t.sweepDelay, 200);
  assert.equal(t.sweepMs, 698); // round(640 × 1.09)
  assert.equal(t.flashDelay, 1040);
});

test('로딩이 즉시 끝나면 약 2.7s 에 끝난다', () => {
  const h = harness();
  h.intro.start();
  assert.ok(h.classes.has('vi-off'));
  h.intro.ready();
  h.advance(80);
  assert.ok(h.classes.has('vi-run'));
  h.advance(1859);
  assert.ok(!h.classes.has('vi-exit'));
  h.advance(1);
  assert.ok(h.classes.has('vi-exit'));
  h.advance(200 + 698 + 60 - 1);
  assert.equal(h.calls.complete, 0);
  h.advance(1);
  assert.equal(h.calls.complete, 1);
  assert.equal(h.el.removed, true);
  assert.equal(h.pending.size, 0);
});

test('ready 가 늦으면 그 시점에 비행으로 넘어간다', () => {
  const h = harness();
  h.intro.start();
  h.advance(5000);
  assert.ok(h.classes.has('vi-run'));
  h.intro.ready();
  h.advance(0);
  assert.ok(h.classes.has('vi-exit'));
});

test('ready 가 안 와도 maxWaitMs 뒤에는 나간다', () => {
  const h = harness({ maxWaitMs: 3000 });
  h.intro.start();
  h.advance(3000);
  assert.ok(h.classes.has('vi-exit'));
  h.advance(1000);
  assert.equal(h.calls.complete, 1);
});

test('start 를 두 번 불러도 타이머가 겹치지 않는다', () => {
  const h = harness();
  h.intro.start();
  const n = h.pending.size;
  h.intro.start();
  assert.equal(h.pending.size, n);
});

test('destroy 는 대기 중인 타이머를 모두 지운다', () => {
  const h = harness();
  h.intro.start();
  h.intro.destroy();
  assert.equal(h.pending.size, 0);
  assert.equal(h.el.removed, true);
});

test('모션 줄이기는 400ms 페이드 뒤에 끝난다', () => {
  const h = harness({}, { reducedMotion: true });
  h.intro.start();
  h.intro.ready();
  h.advance(80 + 1860);
  assert.ok(h.classes.has('vi-exit'));
  h.advance(459);
  assert.equal(h.calls.complete, 0);
  h.advance(1);
  assert.equal(h.calls.complete, 1);
});

test('세션당 1회 옵션: 이미 봤으면 바로 건너뛴다', () => {
  const store = { v: '1', getItem() { return this.v; }, setItem(_, v) { this.v = v; } };
  const h = harness({ oncePerSession: true }, { storage: store });
  h.intro.start();
  assert.equal(h.calls.complete, 1);
  assert.equal(h.el.removed, true);
});

test('세션당 1회 옵션: 처음이면 보여 주고 끝에 기록한다', () => {
  const store = { v: null, getItem() { return this.v; }, setItem(_, v) { this.v = v; } };
  const h = harness({ oncePerSession: true }, { storage: store });
  h.intro.start();
  h.intro.ready();
  h.advance(5000);
  assert.equal(h.calls.complete, 1);
  assert.equal(store.v, '1');
});

test('글로우 세기는 gain 변수에 반영된다 (opacity 는 1 에서 멈춘다)', () => {
  const h = harness({ glow: 1.5 });
  h.intro.start();
  assert.equal(h.el.style.props['--vi-gain-opacity'], '1');
  assert.equal(h.el.style.props['--vi-gain-scale'], String(0.7 + 0.3 * 1.5));
});

test('아이콘 생성이 던지면 인트로를 건너뛰고 오버레이를 걷어낸다', () => {
  const icon = { replaceChildren() { throw new Error('unsupported'); } };
  const h = harness({}, { icon, doc: {} });
  h.intro.start();
  assert.equal(h.calls.complete, 1);
  assert.equal(h.el.removed, true);
  assert.equal(h.pending.size, 0);
});

test('maxWaitMs 는 CSS 최후 안전장치(12s)보다 짧게 제한된다', () => {
  const h = harness({ maxWaitMs: 60000 });
  h.intro.start();
  h.advance(10000);
  assert.ok(h.classes.has('vi-exit'));
});
