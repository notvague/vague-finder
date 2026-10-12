const { test } = require('node:test');
const assert = require('node:assert/strict');
const { FeedbackController } = require('../../src/frontend/static/feedback.js');

function element(tag, doc) {
  return {
    tag, ownerDocument: doc, children: [], attrs: {}, listeners: {}, hidden: false, textContent: '',
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener(key, fn) { this.listeners[key] = fn; },
    focus() { this.focused = true; },
    querySelector(tag) {
      for (const child of this.children) {
        if (child.tag === tag) return child;
        const nested = child.querySelector(tag);
        if (nested) return nested;
      }
      return null;
    },
  };
}
function text(node) { return [node.textContent, ...node.children.map(text)].join(' '); }
function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}
const reply = (search = 'first') => ({
  feedback: { search_id: search, token: `${search}-private-token`, retention_days: 30 },
  results: [{ id: 'a', title: '<script>곡</script>', artist: '가수' }, { id: 'b', title: '다른 곡' }],
});
const saved = (body, revision = 1) => ({
  search_id: body.search_id, revision, status: body.action === 'withdraw' ? 'withdrawn' : 'pending',
  selected_song_id: body.selected_song_id, reason: body.reason,
});
function screen(send) {
  const doc = { createElement: tag => element(tag, doc) };
  const root = element('section', doc);
  let keys = 0;
  const calls = [];
  const ui = new FeedbackController(root, {
    key: () => `submission_key_${++keys}`,
    send: async body => { calls.push(structuredClone(body)); return send ? send(body) : saved(body); },
  });
  ui.reset(reply());
  return { ui, root, calls, doc };
}

test('버튼 선택 → 사유 입력 → 저장. 텍스트로 렌더하고 클릭을 재생 핸들러에 전달하지 않는다', async () => {
  const { ui, root, calls, doc } = screen();
  const row = element('span', doc);
  ui.attach(row, reply().results[0]);
  let stopped = false;
  row.children[0].listeners.click({ stopPropagation() { stopped = true; } });
  assert.equal(stopped, true);
  assert.equal(root.hidden, false);
  const input = root.querySelector('textarea');
  assert.equal(input.focused, true);
  input.value = ' 휘파람 소리 '; input.listeners.input();
  assert.match(text(root), /<script>곡<\/script>/);
  assert.equal(root.querySelector('script'), null);
  assert.match(text(root), /30일/);
  await ui.submit();
  assert.equal(calls[0].reason, '휘파람 소리');
  assert.equal(calls[0].search_id, 'first');
  assert.equal(calls[0].token, 'first-private-token');
  assert.match(text(root), /정답을 저장했어요/);
  assert.equal(ui.draft, null);
});

test('저장 중 중복 클릭은 요청 한 번, 응답 유실 재시도는 같은 제출 번호', async () => {
  const pending = deferred();
  const { ui, calls } = screen(() => pending.promise);
  ui.choose('a'); ui.draft.reason = '음색';
  const first = ui.submit();
  await ui.submit();
  ui.choose('b');
  assert.equal(calls.length, 1);
  assert.equal(ui.draft.id, 'a');
  pending.reject(new Error('timeout')); await first;
  assert.equal(ui.draft.reason, '음색');
  await ui.submit();
  assert.equal(calls.length, 2);
  assert.equal(calls[0].submission_id, calls[1].submission_id);
});

for (const failed of [false, true]) {
  test(`새 검색 뒤 옛 ${failed ? '실패' : '성공'} 응답은 새 선택을 덮지 않는다`, async () => {
    const pending = deferred();
    const { ui } = screen(() => pending.promise);
    ui.choose('a'); const old = ui.submit();
    ui.reset(reply('second')); ui.choose('b');
    if (failed) pending.reject(new Error('late'));
    else pending.resolve(saved({ search_id: 'first', selected_song_id: 'a', reason: '' }));
    await old;
    assert.equal(ui.receipt.search_id, 'second');
    assert.equal(ui.draft.id, 'b');
    assert.equal(ui.saved, null);
    assert.equal(ui.message, '');
  });
}

test('정답 변경과 철회는 최신 revision을 보내고 철회에는 곡·사유를 보내지 않는다', async () => {
  const { ui, calls } = screen(body => saved(body, body.expected_revision + 1));
  ui.choose('a'); await ui.submit();
  ui.choose('b'); await ui.submit();
  await ui.submit('withdraw');
  assert.deepEqual(calls.map(c => c.expected_revision), [0, 1, 2]);
  assert.deepEqual(calls.map(c => c.selected_song_id), ['a', 'b', null]);
  assert.equal(calls[2].reason, '');
  assert.equal(ui.saved.status, 'withdrawn');
});

test('충돌은 자동 덮어쓰기 없이 현재 선택을 보여주고 사용자가 다시 제출한다', async () => {
  let attempt = 0;
  const { ui, calls, root } = screen(body => {
    if (++attempt === 1) throw Object.assign(new Error('conflict'), {
      status: 409, current: { revision: 2, status: 'pending', selected_song_id: 'b', reason: '다른 선택' },
    });
    return saved(body, 3);
  });
  ui.choose('a'); ui.draft.reason = '휘파람';
  await ui.submit();
  assert.match(text(root), /다른 곡/);
  assert.equal(ui.draft.reason, '휘파람');
  assert.equal(calls.length, 1);
  await ui.submit();
  assert.equal(calls[1].expected_revision, 2);
  assert.notEqual(calls[1].submission_id, calls[0].submission_id);
});

test('수집 불가·검색 중·후보 밖 선택은 제출 불가, 재검색이 끝나면 새 권한으로 교체', async () => {
  const { ui, root, calls, doc } = screen();
  ui.choose('hidden'); await ui.submit();
  ui.setBusy(true); ui.choose('a'); await ui.submit();
  assert.equal(calls.length, 0);
  ui.reset({ results: reply().results });
  const row = element('span', doc); ui.attach(row, reply().results[0]);
  assert.equal(row.children.length, 0);
  assert.equal(root.hidden, true);
  ui.reset(reply('third')); ui.choose('a'); await ui.submit();
  assert.equal(calls[0].search_id, 'third');
});

test('사유 없이 제출 가능, 너무 긴 사유는 요청하지 않는다', async () => {
  const { ui, calls } = screen();
  ui.choose('a'); ui.draft.reason = 'a'.repeat(1001); await ui.submit();
  assert.equal(calls.length, 0);
  ui.draft.reason = ''; await ui.submit();
  assert.equal(calls[0].reason, '');
});
