// node --test tests/frontend/clarify_flow.test.cjs   (pytest도 tests/frontend/test_frontend_node.py로 함께 돌린다)
//
// 화면(map.js)의 재질문 상태 전이를 **실제 함수로** 실행한다. DOM과 서버 호출만 대체하고 모델·DB는 쓰지 않는다.
//
// map.js는 지도·캔버스까지 한 파일이라 통째로 올릴 수 없다. 필요한 선언만 **이름으로** 꺼내 온다 —
// 주석 문구나 파일 안의 위치에 기대지 않는다. 이름이 바뀌거나 사라지면 무엇이 없는지 말하고 실패한다.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../src/frontend/static/map.js'), 'utf8');

/* 최상위 한 줄 선언(`const NAME = …;` / `let NAME = …;`) */
function declaration(name) {
  const found = new RegExp('^(?:const|let) ' + name + '\\b.*$', 'm').exec(source);
  assert(found, `map.js에 최상위 선언 ${name}이 없다`);
  return found[0];
}

/* 최상위 함수 선언 — 첫 칸에서 닫는 중괄호까지 */
function functionSource(name) {
  const found = new RegExp('^(?:async )?function ' + name + '\\(', 'm').exec(source);
  assert(found, `map.js에 최상위 함수 ${name}이 없다`);
  const end = source.indexOf('\n}', found.index);
  assert(end > found.index, `map.js의 함수 ${name}이 어디서 끝나는지 찾지 못했다`);
  return source.slice(found.index, end + 2);
}

const DECLARATIONS = [
  'BIRTH_YEAR_KEY', 'sessionBirthYear',
  'SEARCH_TIMEOUT_MS', 'SEARCH_PATH', 'CLARIFY_PATH', 'CLARIFY_TIMEOUT_MS',
  'TOP_K', 'CANDIDATE_K', 'MAX_REJECTED', 'MAX_PREVIOUS_CANDIDATES',
  'convo', 'panelMode', 'searchSeq', 'CLARIFY_FAILED_NOTE',
];
const FUNCTIONS = [
  'loadBirthYear', 'saveBirthYear', 'parseBirthYear', 'searchFailureText',
  'isBusy', 'startConvo', 'applyResponse', 'shownIds', 'isBirthYearQuestion', 'dismissBirthYearQuestion',
  'answerBirthYear', 'canRejectMore', 'onReject', 'requestClarification', 'submitTurn', 'runSearch',
];
const PROGRAM = [...DECLARATIONS.map(declaration), ...FUNCTIONS.map(functionSource)].join('\n');

const QUERY = '중학교 때 듣던 노래';

function reply(extra = {}) {
  return {
    analysis: { original_query: QUERY }, results: [{ id: 'a' }],
    candidate_ids: ['a', 'b', 'c'], rejected_ids: [], asked_slots: [], known_slots: [], turn: 1,
    clarify: null, clarify_deferred: true, ...extra,
  };
}

/* 화면 하나. `first`가 첫 검색 응답이다 */
function screen(first = reply()) {
  const storage = new Map();
  const ctx = {
    localStorage: {
      getItem: key => (storage.has(key) ? storage.get(key) : null),
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    explainOpen: new Set(), explainDetailsOpen: new Set(), resultList: { scrollTop: 0 }, clearBtn: {},
    renderPanel() {}, markHits() { return 0; }, setRailClosed() {}, localSearch: () => [],
    statusLog: [],
    requests: [],
  };
  ctx.showStatus = (msg, tone) => ctx.statusLog.push([msg, tone]);
  vm.createContext(ctx);
  vm.runInContext(PROGRAM, ctx);
  const ev = code => vm.runInContext(code, ctx);
  ctx.first = first;
  ev(`startConvo(${JSON.stringify(QUERY)}, first)`);
  return {
    ctx, ev,
    /* 화면 상태를 이쪽 realm의 값으로 */
    state: () => JSON.parse(ev(`JSON.stringify({
      mode: panelMode, turn: convo && convo.turn, rejected: convo && convo.rejectedIds,
      shown: shownIds(), asked: convo && convo.askedSlots, known: convo && convo.knownSlots, answers: convo && convo.answers,
      clarify: convo && convo.clarify, query: convo && convo.query,
    })`)),
    /* 서버 자리. handler(body, kind)가 응답을 돌려주거나 던진다. kind는 'search' | 'clarify' */
    serve(handler) {
      ctx.requestSearch = async (body, requestPath, timeoutMs) => {
        const kind = requestPath === ev('CLARIFY_PATH') ? 'clarify' : 'search';
        ctx.requests.push({ kind, timeoutMs, body: JSON.parse(JSON.stringify(body)) });
        return handler(body, kind);
      };
    },
    kinds: () => ctx.requests.map(r => r.kind),
    lastStatus: () => ctx.statusLog[ctx.statusLog.length - 1],
  };
}

/* 밖에서 끝낼 수 있는 응답 */
function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const QUESTION = { slot: 'genre', question: '어떤 장르에 가까웠나요?', options: [{ value: '발라드', count: 2 }] };

test('연도 답은 검색만 다시 한다 · 질문 조회는 그 뒤의 최신 후보만 보낸다 · 답은 검색 한 번', async () => {
  const s = screen(reply({ clarify: { slot: 'birth_year', options: [] } }));
  // 연령대를 반영한 재검색은 후보가 달라진다
  s.serve((body, kind) => (kind === 'clarify'
    ? { clarify: QUESTION }
    : reply({ results: [{ id: 'b' }], candidate_ids: ['b', 'x', 'y'], rejected_ids: body.rejected_ids || [],
              turn: (body.answers || []).length ? 2 : 1 })));

  await s.ev('answerBirthYear("1998")');
  assert.deepEqual(s.kinds(), ['search']);
  const birth = s.ctx.requests[0].body;
  assert.equal(birth.defer_clarify, true);
  assert.equal(birth.birth_year, 1998);
  assert.deepEqual(birth.prior_analysis, { original_query: QUERY });
  assert.equal(birth.rejected_ids, undefined, '연도 답은 거절이 아니다');
  assert.equal(s.state().turn, 1);

  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['search', 'clarify']);
  const lookup = s.ctx.requests[1];
  assert.equal(lookup.timeoutMs, s.ev('CLARIFY_TIMEOUT_MS'));
  // 질문 조회는 최신 후보와 대화 상태만 보낸다 — 질의·분석·연도는 보내지 않는다(서버가 모르는 필드를 거부한다)
  assert.deepEqual(lookup.body, {
    asked_slots: [], known_slots: [], previous_candidate_ids: ['b', 'x', 'y'], shown_ids: ['b'], rejected_ids: [], turn: 1,
  });
  assert.deepEqual(s.state(), { ...s.state(), mode: 'asking', turn: 1, rejected: [], shown: ['b'] });

  await s.ev('submitTurn({slot:"genre", value:"발라드", skipped:false})');
  assert.deepEqual(s.kinds(), ['search', 'clarify', 'search']);
  const answer = s.ctx.requests[2].body;
  assert.deepEqual(answer.rejected_ids, ['b']);
  assert.deepEqual(answer.answers, [{ slot: 'genre', value: '발라드', skipped: false }]);
  assert.equal(answer.birth_year, 1998, '답변 검색에도 저장된 연도가 함께 간다');
  assert.equal(answer.defer_clarify, true);
  assert.equal(s.state().turn, 2);
});

test('질문이 없으면 거절만 한 번 한다', async () => {
  const s = screen();
  s.serve((body, kind) => (kind === 'clarify' ? { clarify: null } : reply({ results: [{ id: 'b' }], turn: 2 })));
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['clarify', 'search']);
  assert.deepEqual(s.ctx.requests[1].body.rejected_ids, ['a']);
  assert.equal(s.ctx.requests[1].body.turn, 1);
  assert.equal(s.state().turn, 2);
  assert.deepEqual(s.lastStatus(), ['', undefined], '질문이 없는 것은 장애가 아니다 — 안내를 남기지 않는다');
});

for (const [label, error] of [
  ['서버 오류', Object.assign(new Error('search 500'), { status: 500 })],
  ['제한 시간 초과', Object.assign(new Error('timeout'), { name: 'TimeoutError' })],
]) {
  test(`질문 조회 실패(${label}) — 안내를 남기고 거절 검색을 한 번만 한다`, async () => {
    const s = screen();
    s.serve((body, kind) => {
      if (kind === 'clarify') throw error;
      return reply({ results: [{ id: 'b' }], rejected_ids: body.rejected_ids, turn: 2 });
    });
    await s.ev('onReject()');
    assert.deepEqual(s.kinds(), ['clarify', 'search'], '조회를 다시 시도하지 않는다');
    const rejectOnly = s.ctx.requests[1].body;
    assert.deepEqual(rejectOnly.rejected_ids, ['a']);
    assert.deepEqual(rejectOnly.answers, []);
    assert.deepEqual(s.state(), { ...s.state(), mode: 'idle', turn: 2, shown: ['b'], rejected: ['a'] });
    assert.deepEqual(s.lastStatus(), [s.ev('CLARIFY_FAILED_NOTE'), 'warn']);
  });
}

test('질문 조회도 거절 검색도 실패하면 앞의 결과를 그대로 두고, 다시 누를 수 있다', async () => {
  const s = screen();
  let down = true;
  s.serve((body, kind) => {
    if (down) throw Object.assign(new Error('search 500'), { status: 500 });
    return kind === 'clarify' ? { clarify: QUESTION } : reply();
  });
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['clarify', 'search'], '실패가 이어져도 요청은 클릭 한 번에 둘뿐이다');
  assert.deepEqual(s.state(), { ...s.state(), mode: 'idle', turn: 1, shown: ['a'], rejected: [], answers: [] });
  const [message, tone] = s.lastStatus();
  assert.equal(tone, 'warn');
  assert.match(message, /오류로 응답/);

  down = false;
  await s.ev('onReject()');
  assert.equal(s.state().mode, 'asking');
  assert.deepEqual(s.kinds(), ['clarify', 'search', 'clarify']);
});

test('두 번 눌러도 조회는 한 번, 대체 거절 검색 중의 클릭도 무시한다', async () => {
  const s = screen();
  const lookup = deferred(), search = deferred();
  s.serve((body, kind) => (kind === 'clarify' ? lookup.promise : search.promise));

  const first = s.ev('onReject()');
  s.ev('onReject()');
  s.ev('submitTurn(null)');
  assert.deepEqual(s.kinds(), ['clarify']);
  assert.equal(s.state().mode, 'question_loading');

  lookup.reject(new Error('offline'));
  await new Promise(resolve => setImmediate(resolve)); // 대체 거절 검색이 나갈 때까지
  assert.deepEqual(s.kinds(), ['clarify', 'search']);
  assert.equal(s.state().mode, 'loading');
  s.ev('onReject()');
  s.ev('submitTurn(null)');
  assert.deepEqual(s.kinds(), ['clarify', 'search'], '같은 곡을 두 번 거절하지 않는다');

  search.resolve(reply({ results: [{ id: 'b' }], rejected_ids: ['a'], turn: 2 }));
  await first;
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(s.state(), { ...s.state(), mode: 'idle', turn: 2, rejected: ['a'], shown: ['b'] });
});

for (const [label, finish] of [
  ['질문이 늦게 도착', lookup => lookup.resolve({ clarify: QUESTION })],
  ['실패가 늦게 도착', lookup => lookup.reject(new Error('offline'))],
]) {
  test(`조회 중에 새 검색을 하면 ${label}해도 새 결과를 건드리지 않는다`, async () => {
    const s = screen();
    const lookup = deferred();
    s.serve((body, kind) => (kind === 'clarify'
      ? lookup.promise
      : reply({ results: [{ id: 'n1' }], candidate_ids: ['n1', 'n2'] })));

    const pending = s.ev('onReject()');
    await s.ev('runSearch("새 검색")');
    assert.deepEqual(s.state(), { ...s.state(), query: '새 검색', mode: 'idle', shown: ['n1'], turn: 1 });

    finish(lookup);
    await pending;
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(s.kinds(), ['clarify', 'search'], '옛 대화의 거절 검색이 새로 나가지 않는다');
    assert.deepEqual(s.state(), { ...s.state(), query: '새 검색', mode: 'idle', shown: ['n1'], rejected: [], clarify: null });
  });
}

test('대체 거절 검색 중에 새 검색을 하면 늦은 거절 응답이 새 결과를 바꾸지 않는다', async () => {
  const s = screen();
  const stale = deferred();
  s.serve((body, kind) => {
    if (kind === 'clarify') throw new Error('offline');
    if (body.rejected_ids) return stale.promise;       // 옛 대화의 거절 검색
    return reply({ results: [{ id: 'n1' }], candidate_ids: ['n1', 'n2'] });
  });
  const pending = s.ev('onReject()');
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(s.kinds(), ['clarify', 'search']);

  await s.ev('runSearch("새 검색")');
  stale.resolve(reply({ results: [{ id: 'old' }], rejected_ids: ['a'], turn: 2 }));
  await pending;
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(s.state(), { ...s.state(), query: '새 검색', mode: 'idle', shown: ['n1'], rejected: [], turn: 1 });
  assert.notDeepEqual(s.lastStatus(), [s.ev('CLARIFY_FAILED_NOTE'), 'warn'], '옛 대화의 안내가 새 검색에 남지 않는다');
});

test('새 검색이 실패해도 앞 대화의 로딩 표시가 남지 않는다', async () => {
  const s = screen();
  const lookup = deferred();
  s.serve((body, kind) => {
    if (kind === 'clarify') return lookup.promise;
    throw Object.assign(new Error('search 500'), { status: 500 });
  });
  const pending = s.ev('onReject()');
  await s.ev('runSearch("새 검색")');
  lookup.resolve({ clarify: QUESTION });
  await pending;
  assert.equal(s.ev('convo'), null);
  assert.equal(s.ev('panelMode'), 'idle');
});

test('닫은 질문은 다시 조회하지 않고 연다 · 결과가 바뀌면 버리고 새로 조회한다', async () => {
  const s = screen();
  s.serve((body, kind) => (kind === 'clarify'
    ? { clarify: QUESTION }
    : reply({ results: [{ id: 'b' }], candidate_ids: ['b', 'c', 'd'], rejected_ids: body.rejected_ids, turn: 2 })));

  await s.ev('onReject()');
  assert.equal(s.state().mode, 'asking');
  s.ev('panelMode = "idle"');           // "결과로 돌아가기"·Esc가 하는 일
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['clarify'], '같은 결과의 질문을 다시 받아 오지 않는다');
  assert.deepEqual(s.state(), { ...s.state(), mode: 'asking', clarify: QUESTION, turn: 1, rejected: [] });

  await s.ev('submitTurn({slot:"genre", value:"", skipped:true})');
  assert.equal(s.state().clarify, null, '결과·후보가 바뀌면 받아 둔 질문은 버린다');
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['clarify', 'search', 'clarify']);
  assert.deepEqual(s.ctx.requests[2].body.previous_candidate_ids, ['b', 'c', 'd']);
  assert.deepEqual(s.ctx.requests[2].body.shown_ids, ['b']);
  assert.deepEqual(s.ctx.requests[2].body.rejected_ids, ['a']);
});

test('서버가 조회할 것이 없다고 하면(clarify_deferred 거짓) 질문을 조회하지 않고 거절만 한다', async () => {
  const s = screen(reply({ clarify_deferred: false }));
  s.serve(() => reply({ results: [{ id: 'b' }], rejected_ids: ['a'], turn: 2, clarify_deferred: false }));
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['search']);
  assert.deepEqual(s.ctx.requests[0].body.rejected_ids, ['a']);
});

for (const deferredFlag of [true, false]) {
  test(`연도 질문을 닫으면 요청 없이 기억하고, 다음 요청에 실어 보낸다 (clarify_deferred=${deferredFlag})`, async () => {
    const s = screen(reply({ clarify: { slot: 'birth_year', options: [] }, clarify_deferred: deferredFlag }));
    s.serve((body, kind) => (kind === 'clarify' ? { clarify: null } : reply({ turn: 2 })));
    assert.equal(s.state().mode, 'asking');
    s.ev('dismissBirthYearQuestion()');
    assert.deepEqual(s.kinds(), [], '닫기만으로 요청하지 않는다');
    assert.deepEqual(s.state(), { ...s.state(), mode: 'idle', asked: ['birth_year'], turn: 1 });

    await s.ev('onReject()');
    assert.deepEqual(s.kinds(), deferredFlag ? ['clarify', 'search'] : ['search']);
    for (const request of s.ctx.requests) {
      assert.deepEqual(request.body.asked_slots, ['birth_year'], '서버가 연도를 다시 묻지 않게');
    }
  });
}

test('연도 재검색이 실패하면 앞의 결과와 분석을 그대로 둔다', async () => {
  const s = screen(reply({ clarify: { slot: 'birth_year', options: [] } }));
  s.serve(() => { throw new Error('offline'); });
  await s.ev('answerBirthYear("1998")');
  assert.deepEqual(s.state(), { ...s.state(), mode: 'idle', turn: 1, shown: ['a'], rejected: [] });
  assert.equal(s.ev('convo.analysis.original_query'), QUERY);
});

test('검색을 내는 세 경로(첫 검색 · 출생 연도 재검색 · 거절 검색)가 모두 곡 수와 후보 수를 함께 보낸다', async () => {
  const s = screen(reply({ clarify: { slot: 'birth_year', options: [] }, clarify_deferred: false }));
  s.serve(body => reply({ rejected_ids: body.rejected_ids || [], clarify_deferred: false }));

  await s.ev('runSearch("새 검색")');
  await s.ev('answerBirthYear("1998")');
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['search', 'search', 'search']);
  const [first, birth, reject] = s.ctx.requests.map(r => r.body);
  assert.equal(first.prior_analysis, undefined, '첫 검색');
  assert.equal(birth.birth_year, 1998, '출생 연도 재검색');
  assert.deepEqual(reject.rejected_ids, ['a'], '거절 검색');
  for (const body of [first, birth, reject]) {
    assert.equal(body.top_k, 15);
    // 보내지 않으면 서버가 후보를 top_k*3으로 넓혀 검색 깊이가 측정한 구성과 달라진다
    assert.equal(body.candidate_k, 30);
  }
});

/* ---------- 최초 질의에서 이미 말한 성별·장르(known_slots) ---------- */

test('응답의 known_slots를 보관해 질문 조회에 그대로 보낸다 · 연도 답·닫기 뒤에도 남는다', async () => {
  // 사용자 예시 1 — 성별은 말했고 출생 연도 질문이 함께 온다
  const s = screen(reply({ clarify: { slot: 'birth_year', options: [] }, known_slots: ['vocal_gender'] }));
  assert.deepEqual(s.state().known, ['vocal_gender']);
  s.serve((body, kind) => (kind === 'clarify'
    ? { clarify: QUESTION }
    : reply({ results: [{ id: 'b' }], candidate_ids: ['b', 'x', 'y'], known_slots: ['vocal_gender'],
              rejected_ids: body.rejected_ids || [], turn: (body.answers || []).length ? 2 : 1 })));

  await s.ev('answerBirthYear("1998")');
  assert.deepEqual(s.state().known, ['vocal_gender'], '연도 재검색 응답의 값으로 교체된다(같은 원문이라 같다)');
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['search', 'clarify']);
  assert.deepEqual(s.ctx.requests[1].body.known_slots, ['vocal_gender']);
  assert.deepEqual(s.ctx.requests[1].body.asked_slots, [], 'known_slots는 물은 이력에 끼지 않는다');
  assert.deepEqual(s.ctx.requests[1].body.previous_candidate_ids, ['b', 'x', 'y'], '연도 반영 뒤의 최신 후보');
  assert.equal(s.state().clarify.slot, 'genre');

  // 답하고 다시 검색해도 응답의 값이 그대로 실린다
  await s.ev('submitTurn({slot:"genre", value:"발라드", skipped:false})');
  assert.deepEqual(s.state().known, ['vocal_gender']);
  assert.deepEqual(s.ctx.requests[2].body.answers, [{ slot: 'genre', value: '발라드', skipped: false }]);
  assert.equal('known_slots' in s.ctx.requests[2].body, false, '검색 요청에는 되돌려주지 않는다 — 서버가 원문에서 계산한다');
});

test('연도 질문을 닫아도 known_slots는 남고, 질문 조회에 함께 간다', async () => {
  const s = screen(reply({ clarify: { slot: 'birth_year', options: [] }, known_slots: ['vocal_gender'] }));
  s.serve((body, kind) => (kind === 'clarify' ? { clarify: QUESTION } : reply({ turn: 2 })));
  s.ev('dismissBirthYearQuestion()');
  assert.deepEqual(s.state(), { ...s.state(), asked: ['birth_year'], known: ['vocal_gender'] });
  await s.ev('onReject()');
  assert.deepEqual(s.ctx.requests[0].body.asked_slots, ['birth_year']);
  assert.deepEqual(s.ctx.requests[0].body.known_slots, ['vocal_gender']);
});

test('둘 다 말했으면(clarify_deferred 거짓) 조회 없이 거절 검색 한 번이다', async () => {
  // 사용자 예시 2
  const s = screen(reply({ known_slots: ['vocal_gender', 'genre'], clarify_deferred: false }));
  s.serve(body => reply({ results: [{ id: 'b' }], rejected_ids: body.rejected_ids, turn: 2,
                          known_slots: ['vocal_gender', 'genre'], clarify_deferred: false }));
  await s.ev('onReject()');
  assert.deepEqual(s.kinds(), ['search']);
  assert.deepEqual(s.ctx.requests[0].body.rejected_ids, ['a']);
  assert.deepEqual(s.state(), { ...s.state(), mode: 'idle', turn: 2, shown: ['b'], known: ['vocal_gender', 'genre'] });
});

test('새 검색은 known_slots를 응답 값으로 교체한다 — 이전 검색 값과 합치지 않는다', async () => {
  const s = screen(reply({ known_slots: ['vocal_gender', 'genre'], clarify_deferred: false }));
  s.serve((body, kind) => (kind === 'clarify'
    ? { clarify: QUESTION }
    : reply({ results: [{ id: 'n1' }], candidate_ids: ['n1', 'n2'], known_slots: ['genre'] })));
  await s.ev('runSearch("새 검색")');
  assert.deepEqual(s.state().known, ['genre']);
  await s.ev('onReject()');
  assert.deepEqual(s.ctx.requests[1].body.known_slots, ['genre']);
  // 아무것도 말하지 않은 응답이면 빈 목록
  s.serve(() => reply({ results: [{ id: 'm1' }] }));
  await s.ev('runSearch("또 새 검색")');
  assert.deepEqual(s.state().known, []);
  // 구버전 서버 응답(필드 없음)도 빈 목록
  s.serve(() => { const r = reply(); delete r.known_slots; return r; });
  await s.ev('runSearch("옛 서버")');
  assert.deepEqual(s.state().known, []);
});

test('조회 중에 새 검색을 하면 늦은 질문이 새 검색의 known_slots를 건드리지 않는다', async () => {
  const s = screen(reply({ known_slots: ['vocal_gender'] }));
  const lookup = deferred();
  s.serve((body, kind) => (kind === 'clarify'
    ? lookup.promise
    : reply({ results: [{ id: 'n1' }], candidate_ids: ['n1', 'n2'], known_slots: ['genre'] })));
  const pending = s.ev('onReject()');
  assert.deepEqual(s.ctx.requests[0].body.known_slots, ['vocal_gender']);
  await s.ev('runSearch("새 검색")');
  lookup.resolve({ clarify: QUESTION });
  await pending;
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(s.state(), { ...s.state(), query: '새 검색', known: ['genre'], clarify: null, mode: 'idle' });
  assert.deepEqual(s.kinds(), ['clarify', 'search']);
});

test('한 쪽 15곡을 두 번 거절하면 한도에 닿는다', async () => {
  const topK = s => s.ev('TOP_K');
  const page = (n, size) => Array.from({ length: size }, (_, i) => ({ id: `p${n}-${i}` }));
  const s = screen(reply({ results: page(0, 15), clarify_deferred: false }));
  assert.equal(topK(s), 15);
  let n = 0;
  s.serve(body => reply({ results: page(++n, 15), rejected_ids: body.rejected_ids, turn: n + 1, clarify_deferred: false }));

  assert.equal(s.ev('canRejectMore()'), true);
  await s.ev('onReject()');
  assert.equal(s.ev('canRejectMore()'), true, '두 번째 거절까지 한도 안이다');
  await s.ev('onReject()');
  assert.equal(s.state().rejected.length, 30);
  assert.equal(s.ev('canRejectMore()'), false, '세 번째는 한도(MAX_REJECTED)를 넘는다');
});
