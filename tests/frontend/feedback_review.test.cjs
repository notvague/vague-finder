// 검토 화면(review.js)의 상태 전이. 모의 DOM으로 돈다 — 실제 브라우저 확인은 docs/feedback.md에 따로 적는다.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ReviewController, remainingText, shortTextWarning } = require('../../src/feedback/review_static/review.js');

function element(tag, doc) {
  return {
    tag, ownerDocument: doc, children: [], attrs: {}, listeners: {}, textContent: '', className: '', value: '', disabled: false,
    // 실제 DOM처럼 노드가 아닌 값(null 포함)은 글자 노드가 된다 — 빈 자리를 그대로 넘기면 화면에 "null"이 찍힌다.
    append(...items) { this.children.push(...items.map(asNode)); },
    replaceChildren(...items) { this.children = items.map(asNode); },
    setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener(key, fn) { this.listeners[key] = fn; },
  };
}
function asNode(item) {
  if (item !== null && typeof item === 'object') return item;
  return { tag: '#text', children: [], attrs: {}, listeners: {}, textContent: String(item) };
}
function walk(node, visit) { visit(node); for (const child of node.children) walk(child, visit); }
function find(node, test) { let found = null; walk(node, n => { if (!found && test(n)) found = n; }); return found; }
function all(node, test) { const out = []; walk(node, n => { if (test(n)) out.push(n); }); return out; }
function text(node) { return [node.textContent, ...node.children.map(text)].join(' '); }
const byAttr = (key, value) => n => n.attrs[key] === value;
function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}
function conflict(message, code, current, status = 409) {
  return Object.assign(new Error(message), { status, code, current });
}

const XSS = '<img src=x onerror=alert(1)><script>alert(2)</script>';
const item = (id, extra = {}) => ({
  search_id: id, user_status: 'pending', review_state: 'pending', review_state_label: '대기', revision: 1, review_version: 0,
  selected_song: { id: 'a', title: '곡 A', artist: '가수 A', rank: 3 }, has_reason: true,
  created_at: '2026-10-01T00:00:00+00:00', expires_at: '2026-10-31T00:00:00+00:00', remaining_seconds: 20 * 86400, index_state: null,
  ...extra,
});
const detail = (id, extra = {}) => ({
  ...item(id), query: '휘파람으로 시작하는 노래', run_found: true, reason: '도입부 휘파람', reusable_text: '', review: null,
  context: { turn: 2, answers: [{ slot: 'vocal_gender', value: '여성', skipped: false }], asked_slots: ['vocal_gender'],
    rejected: [{ id: 'x', title: '거절한 곡', artist: '가수 X' }], release_era: { start_year: 2010, end_year: 2013 }, analysis: {} },
  shown: [], current_song: { id: 'a', found: true, title: '곡 A', artist: '가수 A', album: '앨범', release_date: '2012-03-01', genre: '발라드' },
  review_history: [], user_revisions: 1, reviewable: true,
  ...extra,
});

function screen(overrides = {}) {
  const doc = { createElement: tag => element(tag, doc) };
  const root = element('main', doc);
  const calls = { list: [], detail: [], review: [] };
  let clock = 1_000_000;
  const api = {
    session: async () => ({ reviewer: '정현(로컬)', reviewer_verified: false, csrf_token: 'page-token', limits: { reusable_text: 500, note: 1000 } }),
    status: async () => ({ ttl: { search_runs: 'ready', search_feedback: 'ready' }, ttl_ready: true, index: { approved: 0, indexed: 0, failed: 0 }, worker: null }),
    list: async (state, page) => { calls.list.push([state, page]); return { items: [item('one'), item('two')], total: 45 }; },
    detail: async id => { calls.detail.push(id); return detail(id); },
    review: async (id, body, token) => {
      calls.review.push({ id, body: structuredClone(body), token });
      return detail(id, { review_state: body.state, review_version: body.expected_review_version + 1, reusable_text: body.reusable_text.trim(),
        review: { state: body.state, note: body.note, reviewed_by: '정현(로컬)', reviewed_at: '2026-10-11T00:00:00+00:00', version: 1, source_revision: 1 } });
    },
    ...overrides,
  };
  const ui = new ReviewController(root, { api, now: () => clock });
  return { ui, root, calls, advance: seconds => { clock += seconds * 1000; } };
}
const type = (root, role, value) => { const box = find(root, byAttr('data-role', role)); box.value = value; box.listeners.input(); };
const click = (root, key, value) => find(root, byAttr(key, value)).listeners.click();

test('목록 → 상세 → 승인. 화면이 본 제출 판·검토 판과 화면 토큰을 함께 보낸다', async () => {
  const { ui, root, calls } = screen();
  await ui.start();
  assert.deepEqual(calls.list, [['pending', 1]]);
  assert.match(text(root), /로컬 설정값 · 확인된 신원이 아닙니다/);
  assert.match(text(root), /곡 A — 가수 A \(표시 3위\)/);
  click(root, 'data-id', 'one');
  await Promise.resolve(); await Promise.resolve();
  assert.match(text(root), /휘파람으로 시작하는 노래/);
  assert.match(text(root), /가수 성별: 여성/);
  assert.match(text(root), /거절한 곡 1개/);
  assert.match(text(root), /2010~2013년/);
  assert.match(text(root), /검토·승인으로 기간이 늘지 않습니다/);
  type(root, 'reusable-text', '  휘파람 전주  ');
  type(root, 'review-note', '확인');
  await ui.save('approved');
  assert.deepEqual(calls.review[0], { id: 'one', token: 'page-token',
    body: { state: 'approved', reusable_text: '  휘파람 전주  ', note: '확인', expected_revision: 1, expected_review_version: 0 } });
  assert.match(text(root), /승인\(으\)로 저장했습니다/);
  assert.equal(ui.detail.review_version, 1);
  assert.equal(calls.list.length, 2, '저장 뒤 목록을 다시 읽는다');
});

test('빈 자리가 "null" 글자로 찍히지 않는다(실제 브라우저에서 발견)', async () => {
  const { ui, root } = screen({ list: async () => ({ items: [], total: 0 }) });
  await ui.start();
  assert.doesNotMatch(text(root), /null|undefined/);
  ui.list = { items: [item('one')], total: 1 }; ui.renderList();   // 목록이 있고 오류가 없는 쪽
  await ui.open('one');
  assert.doesNotMatch(text(root), /null|undefined/);
  assert.deepEqual(all(root, n => n.tag === '#text'), []);
});

test('빈 문장으로는 승인 요청을 보내지 않는다. 보류·제외는 문장 없이 저장된다', async () => {
  const { ui, root, calls } = screen();
  await ui.start(); await ui.open('one');
  type(root, 'reusable-text', '   ');
  await ui.save('approved');
  assert.equal(calls.review.length, 0);
  assert.match(text(root), /재사용할 단서를 적어야 합니다/);
  await ui.save('on_hold');
  await ui.save('rejected');
  assert.deepEqual(calls.review.map(call => call.body.state), ['on_hold', 'rejected']);
  assert.equal(calls.review[1].body.expected_review_version, 1, '앞 저장이 올린 검토 판으로 다음 저장을 보낸다');
});

test('수정 충돌: 최신 내용을 보여주고 작성 중이던 입력은 지우지 않는다', async () => {
  const latest = detail('one', { revision: 2, selected_song: { id: 'b', title: '바뀐 곡', artist: '가수 B', rank: 1 }, reason: '다른 곡이었어요' });
  let attempt = 0;
  const { ui, root, calls } = screen({
    review: async (id, body) => {
      calls.review.push({ id, body: structuredClone(body) });
      if (++attempt === 1) throw conflict('사용자가 선택을 바꿨습니다. 최신 내용을 확인해 주세요.', 'stale_revision', latest);
      return detail(id, { revision: 2, review_state: 'approved', review_version: 1 });
    },
  });
  await ui.start(); await ui.open('one');
  type(root, 'reusable-text', '작성 중이던 단서');
  type(root, 'review-note', '작성 중이던 메모');
  await ui.save('approved');
  assert.match(text(root), /바뀐 곡 — 가수 B/);
  assert.match(text(root), /다른 곡이었어요/);
  assert.match(text(root), /작성 중이던 내용은 그대로 두었습니다/);
  assert.equal(find(root, byAttr('data-role', 'reusable-text')).value, '작성 중이던 단서');
  assert.equal(find(root, byAttr('data-role', 'review-note')).value, '작성 중이던 메모');
  await ui.save('approved');                      // 최신 내용을 확인한 뒤 다시 누른다
  assert.equal(calls.review[1].body.expected_revision, 2, '다시 저장할 때는 최신 제출 판을 보낸다');
  assert.equal(calls.review[1].body.reusable_text, '작성 중이던 단서');
});

test('철회·만료된 사례는 버튼과 입력이 잠기고 요청을 보내지 않는다', async () => {
  for (const extra of [{ user_status: 'withdrawn', reviewable: false }, { reviewable: false, remaining_seconds: 0 }]) {
    const { ui, root, calls } = screen({ detail: async id => detail(id, extra) });
    await ui.start(); await ui.open('one');
    for (const state of ['approved', 'on_hold', 'rejected', 'pending']) {
      assert.equal(find(root, byAttr('data-action', state)).disabled, true);
    }
    assert.equal(find(root, byAttr('data-role', 'reusable-text')).disabled, true);
    await ui.save('approved');
    assert.equal(calls.review.length, 0);
    assert.match(text(root), extra.user_status ? /철회한 사례/ : /보관 기간이 지난 사례/);
  }
});

test('화면을 열어 둔 사이 보관 기간이 지나면 저장을 막는다', async () => {
  const { ui, root, calls, advance } = screen({ detail: async id => detail(id, { remaining_seconds: 30 }) });
  await ui.start(); await ui.open('one');
  type(root, 'reusable-text', '단서');
  advance(31);
  await ui.save('approved');
  assert.equal(calls.review.length, 0);
  assert.match(text(root), /보관 기간이 지난 사례/);
});

test('저장 중 서버가 만료·철회를 알리면 입력은 남기고 버튼을 잠근다', async () => {
  const withdrawn = detail('one', { user_status: 'withdrawn', reviewable: false, revision: 2 });
  for (const error of [conflict('보관 기간이 지난 사례는 검토할 수 없습니다.', 'expired', null, 410),
    conflict('사용자가 철회한 사례는 검토할 수 없습니다.', 'withdrawn', withdrawn)]) {
    const { ui, root } = screen({ review: async () => { throw error; } });
    await ui.start(); await ui.open('one');
    type(root, 'reusable-text', '남겨 둘 단서');
    await ui.save('approved');
    assert.equal(find(root, byAttr('data-action', 'approved')).disabled, true);
    assert.equal(find(root, byAttr('data-role', 'reusable-text')).value, '남겨 둘 단서');
    assert.match(text(root), /검토할 수 없습니다/);
  }
});

test('저장 실패(연결 오류)는 입력을 유지하고 같은 값으로 다시 보낼 수 있다', async () => {
  let attempt = 0;
  const { ui, root, calls } = screen({
    review: async (id, body) => {
      calls.review.push(structuredClone(body));
      if (++attempt === 1) throw Object.assign(new Error('저장소에 연결하지 못했습니다.'), { status: 503 });
      return detail(id, { review_state: 'approved', review_version: 1 });
    },
  });
  await ui.start(); await ui.open('one');
  type(root, 'reusable-text', '단서');
  await ui.save('approved');
  assert.match(text(root), /입력은 그대로 있습니다/);
  assert.equal(find(root, byAttr('data-action', 'approved')).disabled, false);
  await ui.save('approved');
  assert.deepEqual(calls.review[0], calls.review[1], '응답을 잃은 뒤의 재시도는 같은 판·같은 내용이다');
});

test('저장 중 중복 클릭은 요청 한 번. 다른 사례를 열면 늦은 응답이 그 화면을 덮지 않는다', async () => {
  const pending = deferred();
  const { ui, root, calls } = screen({ review: (id, body) => { calls.review.push(id); return pending.promise; } });
  await ui.start(); await ui.open('one');
  type(root, 'reusable-text', '단서');
  const first = ui.save('approved');
  await ui.save('approved');
  assert.equal(calls.review.length, 1);
  assert.equal(find(root, byAttr('data-action', 'approved')).disabled, true);
  await ui.open('two');
  assert.equal(find(root, byAttr('data-action', 'approved')).disabled, false, '다른 사례의 버튼은 막지 않는다');
  type(root, 'reusable-text', '두 번째 사례의 단서');
  pending.resolve(detail('one', { review_state: 'approved', review_version: 1 }));
  await first;
  assert.equal(ui.detail.search_id, 'two');
  assert.equal(find(root, byAttr('data-role', 'reusable-text')).value, '두 번째 사례의 단서');
});

test('늦게 도착한 목록·상세 응답은 버린다', async () => {
  const slowList = deferred();
  const slowDetail = deferred();
  let lists = 0;
  let details = 0;
  const { ui, root } = screen({
    list: async state => (++lists === 2 ? slowList.promise : { items: [item(`${state}-row`)], total: 1 }),
    detail: async id => (++details === 1 ? slowDetail.promise : detail(id, { query: `${id}의 질의` })),
  });
  await ui.start();
  const stale = ui.setFilter('approved');           // 느린 응답
  await ui.setFilter('rejected');
  slowList.resolve({ items: [item('stale-row')], total: 1 });
  await stale;
  assert.ok(find(root, byAttr('data-id', 'rejected-row')) && !find(root, byAttr('data-id', 'stale-row')));
  const first = ui.open('one');
  await ui.open('two');
  slowDetail.resolve(detail('one', { query: '옛 응답' }));
  await first;
  assert.match(text(root), /two의 질의/);
  assert.doesNotMatch(text(root), /옛 응답/);
});

test('사용자 원문은 글자로만 들어간다 — 마크업 요소를 만들지 않는다', async () => {
  const hostile = detail('one', { query: XSS, reason: XSS, reusable_text: XSS,
    selected_song: { id: 'a', title: XSS, artist: XSS, rank: 1 },
    current_song: { id: 'a', found: true, title: XSS, artist: XSS },
    context: { turn: 1, answers: [{ slot: XSS, value: XSS }], rejected: [{ id: 'x', title: XSS, artist: XSS }] },
    review: { state: 'on_hold', note: XSS, reviewed_by: XSS, version: 1, source_revision: 1 }, review_state: 'on_hold' });
  const { ui, root } = screen({ detail: async () => hostile, list: async () => ({ items: [item('one', { selected_song: hostile.selected_song })], total: 1 }) });
  await ui.start(); await ui.open('one');
  assert.ok(text(root).includes(XSS));
  assert.deepEqual(all(root, n => ['script', 'img', 'iframe'].includes(n.tag)), []);
  walk(root, n => { for (const key of Object.keys(n.attrs)) assert.ok(!key.startsWith('on'), key); assert.equal(n.innerHTML, undefined); });
  assert.equal(find(root, byAttr('data-role', 'reusable-text')).value, XSS);
});

test('상태 필터·쪽 이동과 곡 정보 없음 안내', async () => {
  const { ui, root, calls } = screen({
    detail: async id => detail(id, { current_song: { id: 'a', found: false }, reason: '', context: { turn: 1, answers: [], rejected: [] } }),
  });
  await ui.start();
  await ui.setPage(2);
  await ui.setPage(99);                             // 마지막 쪽(3)으로 묶는다
  await ui.setFilter('on_hold');
  assert.deepEqual(calls.list, [['pending', 1], ['pending', 2], ['pending', 3], ['on_hold', 1]]);
  assert.equal(find(root, byAttr('data-filter', 'on_hold')).attrs['aria-selected'], 'true');
  await ui.open('one');
  assert.match(text(root), /현재 곡 목록에서 찾지 못했습니다/);
  assert.match(text(root), /사유를 적지 않았습니다/);
  assert.match(text(root), /재질문 답변 없음/);
});

test('자동 삭제가 준비되지 않았으면 머리말에 경고한다', async () => {
  const { ui, root } = screen({
    status: async () => ({ ttl: { search_runs: 'ready', search_feedback: 'not_ttl' }, ttl_ready: false, index: { approved: 2, indexed: 1, failed: 1 },
      worker: { last_success_at: null, last_error: 'qdrant_error' }, expired_remaining: { search_runs: 2, search_feedback: 1 } }),
  });
  await ui.start(); await Promise.resolve();
  assert.match(text(root), /자동 삭제\(TTL\) 준비 안 됨 — search_runs 준비됨, search_feedback TTL이 아닌 색인/);
  assert.match(text(root), /최근 오류: qdrant_error/);
  assert.match(text(root), /만료 후 남은 문서 3건/);
});

test('짧은(넓을 수 있는) 단서에는 경고를 보이되 저장은 막지 않는다', async () => {
  const { ui, root, calls } = screen();
  await ui.start(); await ui.open('one');
  const warning = () => find(root, byAttr('data-role', 'text-warning')).textContent;
  assert.equal(warning(), '');
  type(root, 'reusable-text', '슬픈 발라드');
  assert.match(warning(), /넓은 단서는 무관한 검색에도 걸립니다/);
  type(root, 'reusable-text', '도입부에 휘파람이 나오고 중간에 박수 소리가 들어가는 곡');
  assert.equal(warning(), '');
  type(root, 'reusable-text', '슬픈 발라드');
  await ui.save('approved');
  assert.equal(calls.review.length, 1, '판단은 검토자가 한다');
  assert.equal(shortTextWarning('   '), '');
});

test('남은 보관 기간 표시', () => {
  assert.equal(remainingText(0), '만료됨');
  assert.equal(remainingText(-5), '만료됨');
  assert.equal(remainingText(59), '1분 남음');
  assert.equal(remainingText(3 * 3600 + 120), '3시간 2분 남음');
  assert.equal(remainingText(2 * 86400 + 5 * 3600), '2일 5시간 남음');
});
