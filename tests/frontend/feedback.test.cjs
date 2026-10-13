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

test('행의 버튼은 "정답"이다 — 근거와 같은 묶음 규칙을 쓰는 클래스이고 옛 문구는 남지 않는다', () => {
  const { ui, doc } = screen();
  const row = element('span', doc);
  ui.attach(row, reply().results[0]);
  const button = row.children[0];
  assert.equal(button.tag, 'button');
  assert.equal(button.textContent, '정답');
  assert.equal(button.className, 'answerbtn');
  assert.equal(button.type, 'button');
  // 보이는 글자(정답)가 접근성 이름에도 들어 있고, 어느 곡의 버튼인지 구분된다
  assert.match(button.attrs['aria-label'], /^<script>곡<\/script> — 정답/);
  assert.doesNotMatch([button.textContent, button.title, button.attrs['aria-label']].join(' '), /찾던 곡이에요/);
  assert.equal(button.attrs['aria-pressed'], 'false');
});

test('눌린 표시는 고르는 중인 곡에, 없으면 저장된 정답에 한다 · 철회·새 검색이면 지운다', async () => {
  const { ui, doc } = screen(body => saved(body, body.expected_revision + 1));
  const rows = () => {
    // map.js의 renderPanel처럼 결과 행을 새로 만든다 — 붙을 때 상태를 받아야 한다
    ui.setBusy(ui.busy);
    return reply().results.map(track => { const row = element('span', doc); ui.attach(row, track); return row.children[0]; });
  };
  const pressed = buttons => buttons.map(b => b.attrs['aria-pressed']);
  let [a, b] = rows();
  assert.deepEqual(pressed([a, b]), ['false', 'false']);

  a.listeners.click({ stopPropagation() {} });            // a를 고르는 중
  assert.deepEqual(pressed([a, b]), ['true', 'false']);
  await ui.submit();                                       // a 저장
  assert.deepEqual(pressed([a, b]), ['true', 'false']);
  assert.deepEqual(pressed(rows()), ['true', 'false'], '목록을 다시 그려도 저장된 정답에 남는다');

  [a, b] = rows();
  b.listeners.click({ stopPropagation() {} });            // b로 바꾸는 중 — 입력칸이 가리키는 곡에 표시한다
  assert.deepEqual(pressed([a, b]), ['false', 'true']);
  ui.draft = null; ui.render();                            // 닫기 — 저장된 a로 돌아간다
  assert.deepEqual(pressed([a, b]), ['true', 'false']);

  await ui.submit('withdraw');
  assert.deepEqual(pressed([a, b]), ['false', 'false']);
  ui.choose('b'); await ui.submit();
  assert.deepEqual(pressed([a, b]), ['false', 'true']);
  ui.reset(reply('second'));
  assert.deepEqual(pressed(rows()), ['false', 'false'], '새 검색에는 앞 검색의 선택이 남지 않는다');
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

test('수정 한도는 초안을 닫고 새 선택을 막지만 철회·새 검색은 가능하다', async () => {
  const message = '이 검색의 수정 횟수를 모두 사용했습니다.';
  const { ui, root, calls, doc } = screen(body => {
    if (body.search_id === 'first' && body.action === 'select') {
      throw Object.assign(new Error(message), {
        status: 409, code: 'revision_limit',
        current: { search_id: 'first', revision: 20, status: 'pending', selected_song_id: 'a', reason: '' },
      });
    }
    return saved(body, body.expected_revision + 1);
  });
  const row = element('span', doc);
  ui.attach(row, reply().results[1]);
  ui.choose('b'); await ui.submit();
  assert.equal(ui.draft, null);
  assert.equal(ui.pending, null);
  assert.equal(root.querySelector('textarea'), null);
  assert.match(text(root), /이 검색의 수정 횟수를 모두 사용했습니다\./);
  assert.doesNotMatch(text(root), /다시 제출/);
  assert.equal(row.children[0].disabled, true);
  ui.choose('b'); await ui.submit();
  assert.equal(calls.length, 1, '제출을 반복해도 같은 409를 계속 요청하지 않는다');

  ui.setBusy(false); // 결과 목록을 다시 그려도 선택 버튼은 계속 비활성이다.
  const redrawn = element('span', doc);
  ui.attach(redrawn, reply().results[1]);
  assert.equal(redrawn.children[0].disabled, true);
  await ui.submit('withdraw');
  assert.equal(calls[1].expected_revision, 20);
  assert.equal(ui.saved.status, 'withdrawn');
  assert.match(text(root), /철회했어요/);

  ui.reset(reply('second'));
  ui.choose('b'); await ui.submit();
  assert.equal(calls[2].search_id, 'second');
  assert.equal(ui.saved.selected_song_id, 'b');
});

test('HTTP 오류의 revision_limit 코드와 서버 문구를 보존한다', async (t) => {
  const message = '이 검색의 수정 횟수를 모두 사용했습니다.';
  t.mock.method(globalThis, 'fetch', async () => ({
    ok: false, status: 409,
    json: async () => ({ detail: { code: 'revision_limit', message, current: { revision: 20 } } }),
  }));
  await assert.rejects(FeedbackController.send({}), err => {
    assert.equal(err.code, 'revision_limit');
    assert.equal(err.message, message);
    assert.equal(err.current.revision, 20);
    return true;
  });
});

test('사유를 적는 중에 결과 목록이 다시 그려져도 입력칸을 새로 만들지 않는다', () => {
  // map.js의 renderPanel은 그릴 때마다 setBusy(isBusy())를 부른다. 값이 그대로인데 다시 그리면
  // textarea가 새 요소로 바뀌어 포커스와 커서 위치를 잃는다.
  const { ui, root, doc } = screen();
  ui.choose('a');
  const input = root.querySelector('textarea');
  input.value = '도입부 휘파'; input.listeners.input();

  ui.setBusy(false);
  assert.equal(root.querySelector('textarea'), input, '같은 값이면 입력칸이 그대로다');
  // 결과 행은 새로 만들어지므로 버튼은 다시 붙고, 붙을 때 상태를 받는다.
  const row = element('span', doc); ui.attach(row, reply().results[1]);
  assert.equal(row.children[0].disabled, false);

  ui.setBusy(true);
  const disabled = root.querySelector('textarea');
  assert.notEqual(disabled, input, '값이 바뀌면 다시 그린다');
  assert.equal(disabled.disabled, true);
  assert.equal(disabled.value, '도입부 휘파', '적던 내용은 남는다');
  const busyRow = element('span', doc); ui.attach(busyRow, reply().results[1]);
  assert.equal(busyRow.children[0].disabled, true);
});
