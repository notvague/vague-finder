// node --test tests/frontend/explain_panel.test.cjs   (pytest도 tests/frontend/test_frontend_node.py로 함께 돌린다)
//
// 근거 패널(map.js explainPanel)과 실행 기록 줄(renderRunNote)을 **실제 함수로** 그린다. DOM만 대체한다.
//
// 고정하는 것(2026-10-11): 모델이 스스로 쓴 이유(model_notes)는 구형 서버가 문장을 실어 보내도
// 화면에 나타나지 않는다. 서버의 고정 안내(notice)는 기록 없이도 보인다. 검증된 요약·원문 인용은 그대로 보인다.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../src/frontend/static/map.js'), 'utf8');

function declaration(name) {
  const found = new RegExp('^(?:const|let) ' + name + '\\b.*$', 'm').exec(source);
  assert(found, `map.js에 최상위 선언 ${name}이 없다`);
  return found[0];
}

function functionSource(name) {
  const found = new RegExp('^(?:async )?function ' + name + '\\(', 'm').exec(source);
  assert(found, `map.js에 최상위 함수 ${name}이 없다`);
  const end = source.indexOf('\n}', found.index);
  assert(end > found.index, `map.js의 함수 ${name}이 어디서 끝나는지 찾지 못했다`);
  return source.slice(found.index, end + 2);
}

const DECLARATIONS = ['convo', 'runNote', 'panelMode'];
const FUNCTIONS = [
  'makeEl', 'exDelta', 'exGroup', 'exItem', 'exFact', 'exSum', 'exPathDetail',
  'explainMix', 'explainLyric', 'explainEvidence', 'explainPanel', 'explainClues', 'explainRecord',
  'toggleExplain', 'renderRunNote', 'applyResponse', 'isBirthYearQuestion',
  'explainToggle', 'resultActions',
];
const PROGRAM = [...DECLARATIONS.map(declaration), ...FUNCTIONS.map(functionSource)].join('\n');

/* 아주 작은 DOM. 텍스트를 모아 "무엇이 보이는가"만 묻는다 */
function node(tag) {
  const classes = new Set();
  const self = {
    tag, className: '', textContent: '', children: [], hidden: false, attrs: {},
    classList: {
      add(c) { classes.add(c); },
      toggle(c, on) { if (on) classes.add(c); else classes.delete(c); },
      // 실제 DOM처럼 className으로 붙인 클래스도 본다(makeEl은 className에 쓴다)
      contains(c) { return classes.has(c) || self.className.split(/\s+/).includes(c); },
    },
    append(...items) { this.children.push(...items); },
    setAttribute(k, v) { this.attrs[k] = v; },
    addEventListener(type, fn) { (this.listeners ||= {})[type] = fn; },
  };
  return self;
}
/* 태그·클래스로 후손을 찾는다 — "접힌 상세 안에 있는가"를 물을 때 */
function find(n, pred, out = []) {
  if (!n || typeof n === 'string' || n.isText) return out;
  if (pred(n)) out.push(n);
  for (const c of n.children) find(c, pred, out);
  return out;
}
function hasClass(n, c) { return n.className && n.className.split(/\s+/).includes(c); }
function allText(n) {
  if (typeof n === 'string') return n;
  if (n.isText) return n.textContent;
  // 글자 노드와 <mark>가 섞인 인용 줄은 붙여 읽고, 요소끼리는 띄어 읽는다
  const mixed = n.children.some((c) => c.isText);
  return [n.textContent, ...n.children.map(allText)].filter(Boolean).join(mixed ? '' : ' ');
}

function realm() {
  const runNoteEl = node('p');
  const ctx = {
    document: {
      createElement: (tag) => node(tag),
      createTextNode: (text) => ({ isText: true, textContent: String(text) }),
      getElementById: () => runNoteEl,
    },
    // applyResponse가 부르는 화면 갱신 — 여기서는 상태만 본다
    explainOpen: new Set(), explainDetailsOpen: new Set(),
    markHits() { return 0; }, renderPanel() {}, resultList: { scrollTop: 0 },
    railSyncs: 0, syncRailMetrics() { ctx.railSyncs += 1; },
  };
  vm.createContext(ctx);
  vm.runInContext(PROGRAM, ctx);
  return { ctx, ev: (code) => vm.runInContext(code, ctx), runNoteEl };
}

const FABRICATED = '뮤직비디오에 옥상에서 의자를 던지는 장면이 나온다';
const NOTICE = '말씀하신 뮤직비디오 장면은 확인할 자료가 없어, 다른 단서로 찾은 후보를 보여드려요.';

/* 구형 서버가 모델 문장을 실어 보낸 곡 설명 */
function legacyExplain(extra = {}) {
  return {
    song_id: '602193305',
    summary: '텍스트(의미+키워드) 경로 1위 (+0.0164) · 말한 성별과 일치 (+0.0164)',
    paths: [{ path: 'text_hybrid', label: '텍스트(의미+키워드)', rank: 1, delta: 0.0164, detail: '', dropped: false }],
    dropped_paths: [], adjustments: [], order_rules: [],
    path_total: 0.0164, adjustment_total: 0, fused_score: 0.0164, retrieval_score: 0.0164,
    rerank_score: null, rank_before_rerank: 1, rank_after_rerank: 1,
    group_rank_before: null, group_rank_after: null, behind_protected: 0,
    reorder_applied: false, reorder_stage: 'policy_skipped', reorder_stage_label: '리랭킹 생략(정책)',
    lyric_match: null, lyric_match_label: '', lyric_boost_rank_before: null, lyric_boost_rank_after: null,
    score_mix: null,
    model_notes: [FABRICATED],
    model_notes_caption: '모델이 스스로 말한 이유입니다. 실행 기록으로 검증되지 않았고, 최종 순위의 근거와 다를 수 있습니다',
    evidence: [],
    ...extra,
  };
}

test('구형 응답의 model_notes는 근거 패널에 그려지지 않는다 · 요약과 원문 인용은 그대로 보인다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({
    evidence: [{
      song_id: '602193305', source: 'full_lyrics', quote: '옥상', line: '옥상 위로 올라가 수박을 먹어',
      char_start: 0, char_end: 2, data_version: 'v1', matched_phrase: '옥상', kind: 'supplementary',
      kind_label: '순위에 쓰이지 않은 보충 설명',
    }],
  });
  const panel = r.ev('explainPanel(x)');
  const text = allText(panel);
  assert.ok(!text.includes(FABRICATED), `모델 문장이 보인다: ${text}`);
  assert.ok(!text.includes('모델이 쓴 이유'), '모델 이유 묶음 자체가 없어야 한다');
  assert.ok(text.includes('텍스트(의미+키워드) 경로 1위'), '검증된 요약은 보인다');
  assert.ok(text.includes('옥상 위로 올라가 수박을 먹어'), '원문 인용은 보인다');
  assert.ok(text.includes('순위에 쓰이지 않은 보충 설명'), '인용의 성격 라벨도 보인다');
});

test('model_notes가 아예 없는 응답(현행 서버)도 같은 패널을 그린다', () => {
  const r = realm();
  const x = legacyExplain();
  delete x.model_notes;
  delete x.model_notes_caption;
  r.ctx.x = x;
  const text = allText(r.ev('explainPanel(x)'));
  assert.ok(text.includes('텍스트(의미+키워드) 경로 1위'));
});

test('서버 안내(notice)는 기록(explain)이 없어도 실행 기록 줄에 보이고 경고색이 아니다', () => {
  const r = realm();
  r.ev(`convo = { notice: { code: 'mv_scene_unverifiable', message: ${JSON.stringify(NOTICE)} }, explain: null }`);
  r.ev('renderRunNote()');
  assert.equal(r.runNoteEl.hidden, false);
  assert.ok(r.runNoteEl.textContent.includes(NOTICE));
  assert.equal(r.runNoteEl.classList.contains('warn'), false);
});

test('안내와 기록이 함께 오면 안내가 먼저, 단계 이름이 뒤따른다 · 안내가 없으면 문장이 남지 않는다', () => {
  const r = realm();
  r.ev(`convo = {
    notice: { code: 'mv_scene_unverifiable', message: ${JSON.stringify(NOTICE)} },
    explain: { reorder_stage: 'policy_skipped', reorder_stage_label: '리랭킹 생략(정책)', reranker: 'GeminiListwiseReranker',
               rerank_calls: 0, rerank_calls_applied: 0, rerank_error: '', analysis_fallback: false, failed_paths: [] },
  }`);
  r.ev('renderRunNote()');
  const first = r.runNoteEl.textContent;
  assert.ok(first.startsWith(NOTICE), first);
  assert.ok(first.includes('리랭킹 생략(정책)'));
  assert.ok(!first.includes('번 불러'), '부르지 않은 호출 수는 적지 않는다');

  // 다음 검색에는 안내가 없다 — 앞 안내가 남으면 안 된다
  r.ev(`convo = { notice: null, explain: { reorder_stage: 'model_applied', reorder_stage_label: '리랭킹 적용', reranker: 'GeminiListwiseReranker',
        rerank_calls: 1, rerank_calls_applied: 1, rerank_error: '', analysis_fallback: false, failed_paths: [] } }`);
  r.ev('renderRunNote()');
  assert.ok(!r.runNoteEl.textContent.includes(NOTICE));
  assert.ok(r.runNoteEl.textContent.includes('리랭킹 적용'));
});

test('응답을 받으면 안내는 응답 값으로 교체된다 — 안내 없는 다음 응답이 앞 안내를 지운다', () => {
  const r = realm();
  r.ev('convo = { query: "q", answers: [] }');
  r.ev(`applyResponse({ results: [], notice: { code: 'mv_scene_unverifiable', message: ${JSON.stringify(NOTICE)} } })`);
  assert.equal(r.ev('convo.notice.message'), NOTICE);
  r.ev('applyResponse({ results: [] })');
  assert.equal(r.ev('convo.notice'), null, '앞 결과의 안내가 남았다');
  r.ev('renderRunNote()');
  assert.equal(r.runNoteEl.hidden, true);
});

test('map.js에 모델 이유를 그리던 함수가 남아 있지 않다', () => {
  assert.ok(!/function explainModelNotes\(/.test(source));
  assert.ok(!/exmodelnote/.test(functionSource('explainPanel')));
  assert.ok(!/exmodelnote|x\.model_notes/.test(functionSource('explainRecord')));
});

/* ---------- 사용자용 설명 (2026-10-12) ---------- */

const USER_SUMMARY = '남성 보컬 단서는 곡 정보와 일치해요. 인디 여부, 말씀하신 뮤직비디오 장면은 확인할 자료가 없어요.';
const SCRIPTY = '<script>alert("x")</script> & "따옴표" <b>굵게</b>';
function userExplanation(extra = {}) {
  return {
    summary: USER_SUMMARY,
    clue_checks: [
      { slot: 'vocal_gender', clue: '남성 보컬', clue_origin: 'query', observed: '등록된 보컬 정보: 남성',
        status: 'match', status_label: '일치', source: 'track.vocal_gender', ranking_basis: [], basis_label: '순위 반영 기록 없음' },
      { slot: 'genre', clue: '인디', clue_origin: 'query', observed: '등록 장르는 댄스이며 인디 여부를 확인할 자료는 없음',
        status: 'unverified', status_label: '미확인', source: 'track.genre', ranking_basis: [], basis_label: '순위 반영 기록 없음' },
      { slot: 'mv_scene', clue: '뮤비에 의자가 하늘에 떠있는 장면이 있어', clue_origin: 'query',
        observed: '해당 MV 장면을 확인할 자료 없음(자료 없음은 장면이 없다는 뜻이 아님)',
        status: 'unverified', status_label: '미확인', source: 'rerank_policy.MV_SCENE_EVIDENCE_FIELDS (없음)',
        ranking_basis: [{ kind: 'rerank_skip', code: 'mv_scene_unverifiable', label: '장면을 판단할 자료가 없어 리랭킹 모델을 부르지 않음', delta: null }],
        basis_label: '자료가 없어 리랭킹 생략' },
    ],
    ...extra,
  };
}

test('사용자용 설명이 있으면 짧은 설명 → 단서 대조 → 접힌 점수 상세 순서로 그린다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({ user_explanation: userExplanation(), model_notes: [FABRICATED] });
  const panel = r.ev('explainPanel(x)');
  const box = panel.children[0];
  const kinds = box.children.map((c) => `${c.tag}.${c.className}`);
  assert.deepEqual(kinds, ['p.exsummary exuser', 'table.exclues', 'details.exdetails'], kinds.join(' | '));
  assert.equal(box.children[0].textContent, USER_SUMMARY);

  // 단서 표: 세 행, 상태는 글자로 보인다
  const rows = find(panel, (n) => n.tag === 'tr' && hasClass(n, 'exclue'));
  assert.equal(rows.length, 3);
  const statuses = rows.map((tr) => allText(tr.children[2]));
  assert.deepEqual(statuses, ['일치', '미확인', '미확인']);
  assert.ok(hasClass(rows[0], 'exclue-match') && hasClass(rows[1], 'exclue-unverified'));
  assert.ok(allText(rows[1]).includes('인디 여부를 확인할 자료는 없음'));
  assert.ok(allText(rows[2]).includes('자료가 없어 리랭킹 생략'), '순위 기여 문구는 서버 것을 그대로 쓴다');
  assert.ok(allText(rows[0]).includes('순위 반영 기록 없음'), '곡 정보 일치와 점수 반영은 따로 적힌다');

  // 접힌 상세: 기본 닫힘(open 속성 없음), 토글 문구, 기존 숫자 장부가 그 안에 있다
  const details = box.children[2];
  assert.equal(details.attrs.open, undefined);
  assert.equal(details.children[0].tag, 'summary');
  assert.equal(details.children[0].textContent, '점수 상세 보기');
  const inside = allText(details);
  assert.ok(inside.includes('텍스트(의미+키워드) 경로 1위'), '기록 요약은 상세 안에 남는다');
  assert.ok(inside.includes('경로 기여') && inside.includes('+0.0164'), '숫자는 상세 안에 보존된다');
  assert.ok(details.listeners && details.listeners.toggle, '펼치면 레일 높이를 다시 잰다');
  // 모델 문장은 어디에도 없다
  assert.ok(!allText(panel).includes(FABRICATED));
});

test('user_explanation이 없는 구형 응답은 예전 배치 그대로 — 기록 요약과 표가 바로 보이고 접힘이 없다', () => {
  const r = realm();
  const x = legacyExplain();
  delete x.user_explanation;
  r.ctx.x = x;
  const panel = r.ev('explainPanel(x)');
  const box = panel.children[0];
  assert.equal(box.children[0].className, 'exsummary');
  assert.ok(!box.children.some((c) => c.tag === 'details'));
  assert.ok(allText(panel).includes('텍스트(의미+키워드) 경로 1위'));
});

test('user_explanation이 null이거나 빈 표여도 오류 없이 그린다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({ user_explanation: null });
  assert.ok(allText(r.ev('explainPanel(x)')).includes('텍스트(의미+키워드) 경로 1위'));
  r.ctx.x = legacyExplain({ user_explanation: { summary: '일부 단서가 비슷한 후보예요.', clue_checks: [] } });
  const panel = r.ev('explainPanel(x)');
  const kinds = panel.children[0].children.map((c) => c.tag);
  assert.deepEqual(kinds, ['p', 'details'], '빈 표는 그리지 않는다');
});

test('기록 배열이 빠진 구형 응답도 오류 없이 그린다', () => {
  const r = realm();
  const x = legacyExplain({ user_explanation: userExplanation() });
  delete x.order_rules; delete x.paths; delete x.dropped_paths; delete x.adjustments; delete x.evidence;
  r.ctx.x = x;
  assert.ok(allText(r.ev('explainPanel(x)')).includes(USER_SUMMARY));
});

test('질의·곡 설명의 특수문자는 텍스트로만 그려진다 · 긴 문장도 잘리지 않는다', () => {
  const r = realm();
  const long = '아주 '.repeat(80) + '긴 단서';
  r.ctx.x = legacyExplain({
    user_explanation: userExplanation({
      summary: SCRIPTY,
      clue_checks: [{ slot: 'lyrics', clue: SCRIPTY, clue_origin: 'analysis', observed: long,
                      status: 'partial', status_label: '부분 일치', source: 'x', ranking_basis: [], basis_label: '검색 점수에 반영' }],
    }),
  });
  const panel = r.ev('explainPanel(x)');
  const summary = panel.children[0].children[0];
  assert.equal(summary.textContent, SCRIPTY, '문장은 textContent 그대로다(innerHTML 아님)');
  const row = find(panel, (n) => n.tag === 'tr' && hasClass(n, 'exclue'))[0];
  assert.equal(row.children[0].textContent, SCRIPTY);
  assert.ok(allText(row.children[1]).includes(long));
  assert.equal(allText(row.children[2]), '부분 일치');
  assert.ok(!/innerHTML/.test(functionSource('explainClues')) && !/innerHTML/.test(functionSource('explainPanel')));
});

test('근거 패널을 그릴 때 네트워크 요청이 나가지 않는다', () => {
  const r = realm();
  let fetches = 0;
  r.ctx.fetch = () => { fetches += 1; return Promise.resolve(); };
  r.ctx.x = legacyExplain({ user_explanation: userExplanation() });
  r.ev('explainPanel(x)');
  assert.equal(fetches, 0);
  assert.ok(!/fetch\(|requestSearch\(/.test(functionSource('explainPanel') + functionSource('explainClues') + functionSource('explainRecord')));
});

test('UFE-10: 목록을 다시 그려도 펼쳐 둔 점수 상세는 펼친 채이고, 새 응답이 오면 비워진다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({ user_explanation: userExplanation() });
  const first = r.ev('explainPanel(x)');
  const details = first.children[0].children[2];
  assert.equal(details.open, false, '처음엔 접혀 있다');
  // 사용자가 펼쳤다(네이티브 toggle 이벤트)
  details.open = true;
  details.listeners.toggle();
  assert.ok(r.ctx.explainDetailsOpen.has('602193305'));
  // 질문이 뜨거나 닫혀 목록을 다시 그렸다 — 같은 곡의 상세는 펼친 채로 돌아온다
  const again = r.ev('explainPanel(x)');
  assert.equal(again.children[0].children[2].open, true);
  // 다시 접으면 집합에서 빠진다
  const d2 = again.children[0].children[2];
  d2.open = false; d2.listeners.toggle();
  assert.ok(!r.ctx.explainDetailsOpen.has('602193305'));

  // 새 검색 응답 — 앞 목록의 펼침 상태(근거·상세 둘 다)는 남지 않는다 (UFE-T3)
  r.ctx.explainOpen.add('602193305'); r.ctx.explainDetailsOpen.add('602193305');
  r.ev('convo = { query: "q", answers: [] }');
  r.ev('applyResponse({ results: [] })');
  assert.equal(r.ctx.explainOpen.size, 0);
  assert.equal(r.ctx.explainDetailsOpen.size, 0);
});

test('점수 상세를 펼치거나 접을 때마다 레일 높이를 다시 잰다 — 좁은 화면에서 지도가 따라 내려가야 한다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({ user_explanation: userExplanation() });
  const details = r.ev('explainPanel(x)').children[0].children[2];
  assert.equal(r.ctx.railSyncs, 0, '그리기만 해서는 재지 않는다');
  details.open = true; details.listeners.toggle();
  assert.equal(r.ctx.railSyncs, 1, '펼칠 때');
  details.open = false; details.listeners.toggle();
  assert.equal(r.ctx.railSyncs, 2, '접을 때');
});

test('근거를 닫았다가 다시 열면 점수 상세는 기본(접힘)으로 돌아간다 · 닫을 때 패널 줄만 지운다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({ user_explanation: userExplanation() });
  const id = '602193305';
  // 결과 줄(li)과 그 뒤에 끼어드는 패널 줄을 흉내 낸다
  const row = { nextElementSibling: null, after(panel) { this.nextElementSibling = panel; panel.remove = () => { this.nextElementSibling = null; }; } };
  const btn = { attrs: {}, closest: () => row, setAttribute(k, v) { this.attrs[k] = v; } };
  r.ctx.row = row; r.ctx.btn = btn;

  r.ev(`toggleExplain('${id}', btn, x)`);
  assert.equal(btn.attrs['aria-expanded'], 'true');
  const details = row.nextElementSibling.children[0].children[2];
  details.open = true; details.listeners.toggle();           // 사용자가 상세를 펼쳤다
  assert.ok(r.ctx.explainDetailsOpen.has(id));

  r.ev(`toggleExplain('${id}', btn, x)`);                      // 근거를 닫는다
  assert.equal(btn.attrs['aria-expanded'], 'false');
  assert.equal(row.nextElementSibling, null);
  assert.ok(!r.ctx.explainOpen.has(id) && !r.ctx.explainDetailsOpen.has(id));

  r.ev(`toggleExplain('${id}', btn, x)`);                      // 다시 연다 — 상세는 접혀 있다
  assert.equal(row.nextElementSibling.children[0].children[2].open, false);
});

test('UFE-09: 가감점 줄의 서버 표시(note)가 부연으로 보인다 · 없으면 아무것도 붙지 않는다', () => {
  const r = realm();
  r.ctx.x = legacyExplain({
    user_explanation: userExplanation(),
    adjustments: [
      { rule: 'vocal_gender_match', label: '말한 성별과 일치', delta: 0.0164, detail: '남성', slot: 'vocal_gender', value: '남성', note: '정정 전 답(남성) 기준' },
      { rule: 'title_exact', label: '제목 완전일치', delta: 0.08, detail: '', slot: '', value: '', note: '' },
    ],
    adjustment_total: 0.0964,
  });
  const panel = r.ev('explainPanel(x)');
  const rows = find(panel, (n) => n.tag === 'div' && hasClass(n, 'exrow'));
  const gender = rows.find((row) => allText(row).includes('말한 성별과 일치'));
  assert.ok(allText(gender).includes('남성 · 정정 전 답(남성) 기준'));
  const title = rows.find((row) => allText(row).includes('제목 완전일치'));
  assert.ok(!allText(title).includes('기준'));
});

test('정적 파일 버전이 올라갔다', () => {
  const html = fs.readFileSync(path.join(__dirname, '../../src/frontend/static/index.html'), 'utf8');
  // 피드백(#47)이 문자열 버전(?v=feedback-1)으로 바꾼 뒤라 숫자가 아니라 새 문자열로 올린다.
  // 정답 버튼을 근거 옆으로 옮기며 세 파일이 함께 바뀌었다(evidence-1 → answer-1, feedback.js 3 → 4).
  // v59(리랭킹 일부 적용 · 묶음 수 표기)는 map.js만 바뀌어 map.js 버전만 올렸다(answer-1 → rerank-status-1)
  assert.ok(/map\.js\?v=rerank-status-1/.test(html), 'map.js 버전');
  assert.ok(/style\.css\?v=answer-1/.test(html), 'style.css 버전');
  assert.ok(/feedback\.js\?v=4/.test(html), 'feedback.js 버전');
});

test('결과 행 오른쪽 묶음은 근거 → 정답 순서다 · 제출 권한이 없으면 근거만, 둘 다 없으면 묶음이 없다', () => {
  const { FeedbackController } = require('../../src/frontend/static/feedback.js');
  const r = realm();
  // 사유 입력 상자 자리. 여기서는 행의 버튼만 본다 — 상자의 내용은 feedback.test.cjs가 본다
  const root = Object.assign(node('section'), { ownerDocument: r.ctx.document, replaceChildren() {}, querySelector: () => null });
  const feedback = new FeedbackController(root, { send: async () => ({}) });
  r.ctx.VagueFeedback = feedback;
  r.ctx.track = { id: 602193305, title: '젊은 우리 사랑', explain: legacyExplain() };
  r.ctx.bare = { id: 1, title: '기록 없는 곡' };

  // 수집이 꺼진 응답(feedback 없음) — 근거만 선다
  feedback.reset({ results: [r.ctx.track, r.ctx.bare] });
  assert.deepEqual(r.ev(`resultActions('602193305', track)`).children.map((b) => b.textContent), ['근거']);
  assert.equal(r.ev(`resultActions('1', bare)`), null, '버튼이 하나도 없으면 빈 묶음을 행에 넣지 않는다');

  feedback.reset({ feedback: { search_id: 's', token: 't', retention_days: 30 }, results: [r.ctx.track, r.ctx.bare] });
  const actions = r.ev(`resultActions('602193305', track)`);
  assert.ok(hasClass(actions, 'resultactions'));
  assert.deepEqual(actions.children.map((b) => [b.tag, b.className, b.textContent]),
    [['button', 'exbtn', '근거'], ['button', 'answerbtn', '정답']]);
  // 기록이 없는 곡(문자열 검색 대체)에도 정답은 붙는다
  assert.deepEqual(r.ev(`resultActions('1', bare)`).children.map((b) => b.textContent), ['정답']);
  // 정답을 눌러도 지도 카드(행 클릭)는 열리지 않고, 숫자 id도 문자열로 맞춰 고른다
  let stopped = false;
  actions.children[1].listeners.click({ stopPropagation() { stopped = true; } });
  assert.ok(stopped);
  assert.equal(feedback.draft.id, '602193305');
  assert.equal(actions.children[1].attrs['aria-pressed'], 'true');

  // 화면 전체에 옛 문구·옛 클래스가 남지 않았다
  const dir = path.join(__dirname, '../../src/frontend/static');
  for (const name of ['map.js', 'feedback.js', 'style.css', 'index.html']) {
    const text = fs.readFileSync(path.join(dir, name), 'utf8');
    assert.ok(!/찾던 곡이에요|feedbackpick/.test(text), name);
  }
  delete r.ctx.VagueFeedback;
  assert.deepEqual(r.ev(`resultActions('602193305', track)`).children.map((b) => b.textContent), ['근거'], 'feedback.js가 없는 화면');
});

test('UFE-05: 리허설 스크립트(check_demo_faults.py)가 읽는 곡별 설명은 기록 요약이다 — 사용자용 문장이 아니다', () => {
  // 스크립트의 선택자를 그대로 떠 와 실제 패널에 적용한다. 스크립트는 Selenium으로 돌아 여기서 실행하지 않는다
  const py = fs.readFileSync(path.join(__dirname, '../../src/backend/check_demo_faults.py'), 'utf8');
  const line = /summary: panel \? \(\((.+?) \|\| \{\}\)\.textContent/.exec(py);
  assert.ok(line, 'check_demo_faults.py에서 곡별 설명을 읽는 줄을 찾지 못했다');
  const selectors = [...line[1].matchAll(/panel\.querySelector\("([^"]+)"\)/g)].map((m) => m[1]);
  assert.ok(selectors.length >= 1, line[1]);
  const query = (root, selector) => {
    const classes = selector.split('.').filter(Boolean);
    return find(root, (n) => classes.every((c) => hasClass(n, c)))[0];   // 문서 순서의 첫 일치 = querySelector
  };
  const read = (panel) => {
    for (const selector of selectors) {
      const hit = query(panel, selector);
      if (hit) return hit.textContent;
    }
    return '';
  };

  const r = realm();
  const record = '텍스트(의미+키워드) 경로 1위 (+0.0164) → 리랭킹이 실패해 검색 순서 유지';
  r.ctx.x = legacyExplain({ summary: record, user_explanation: userExplanation() });
  const modern = read(r.ev('explainPanel(x)'));
  assert.equal(modern, record, '새 배치에서 사용자용 문장을 읽으면 "리랭킹이 실패"를 찾지 못한다');
  assert.notEqual(modern, USER_SUMMARY);

  const old = legacyExplain({ summary: record });
  delete old.user_explanation;
  r.ctx.x = old;
  assert.equal(read(r.ev('explainPanel(x)')), record, '사용자용 설명이 없는 배치에서도 같은 문장을 읽는다');
});

/* ---------- v59: 리랭킹 일부 적용 · 묶음 수와 실제 호출 수 (2026-10-16) ---------- */

function runExplain(extra = {}) {
  return {
    reorder_stage: 'model_applied', reorder_stage_label: '리랭킹 적용', reorder_stage_note: '',
    reranker: 'GeminiListwiseReranker', rerank_calls: 1, rerank_calls_applied: 1,
    rerank_groups_failed: 0, rerank_groups_budget_exhausted: 0, rerank_model_calls: 1,
    rerank_error: '', analysis_fallback: false, failed_paths: [], ...extra,
  };
}

test('v59: 일부 적용은 단계·서버 문구·묶음 수·예산 소진 묶음·실제 호출 수를 적고 경고색이다 — "N번 불러"는 쓰지 않는다', () => {
  const r = realm();
  r.ev('convo = ' + JSON.stringify({
    notice: null,
    explain: runExplain({
      reorder_stage: 'model_partial', reorder_stage_label: '리랭킹 일부 적용',
      reorder_stage_note: '일부 후보 묶음만 리랭킹했고, 나머지 묶음은 리랭킹 전 후보 순서를 그대로 사용했습니다.',
      rerank_calls: 3, rerank_calls_applied: 1, rerank_groups_failed: 1, rerank_groups_budget_exhausted: 1,
      rerank_model_calls: 2,
    }),
  }));
  r.ev('renderRunNote()');
  const text = r.runNoteEl.textContent;
  assert.ok(text.includes('리랭킹 일부 적용'), text);
  assert.ok(text.includes('나머지 묶음은 리랭킹 전 후보 순서'), text);
  assert.ok(text.includes('후보 묶음 3개 중 1개 리랭킹, 1개는 예산 소진으로 모델 미호출'), text);
  assert.ok(text.includes('모델 호출 2회'), text);
  assert.ok(!text.includes('번 불러'), '묶음 기록 수를 호출 횟수처럼 말하지 않는다');
  assert.ok(!text.includes('3번'), text);
  assert.equal(r.runNoteEl.classList.contains('warn'), true, '일부 폴백은 경고색');
});

test('v59: 모두 적용이면 묶음 문구가 없고 경고색이 아니다 · 호출 수를 세지 않는 백엔드(null)는 호출 횟수를 적지 않는다', () => {
  const r = realm();
  r.ev('convo = ' + JSON.stringify({ notice: null, explain: runExplain({ rerank_model_calls: null }) }));
  r.ev('renderRunNote()');
  const text = r.runNoteEl.textContent;
  assert.ok(text.includes('리랭킹 적용'), text);
  assert.ok(!text.includes('묶음') && !text.includes('모델 호출'), text);
  assert.equal(r.runNoteEl.classList.contains('warn'), false);

  r.ev('convo = ' + JSON.stringify({ notice: null, explain: runExplain({ rerank_model_calls: 1 }) }));
  r.ev('renderRunNote()');
  assert.ok(r.runNoteEl.textContent.includes('모델 호출 1회'));
});

test('v59: 예외 없는 리랭킹 실패(백엔드 내부 폴백)도 경고색이다 · 정책 생략은 여전히 경고가 아니다', () => {
  const r = realm();
  r.ev('convo = ' + JSON.stringify({
    notice: null,
    explain: runExplain({ reorder_stage: 'model_failed', reorder_stage_label: '리랭킹 실패',
                          rerank_calls: 1, rerank_calls_applied: 0, rerank_groups_failed: 1, rerank_model_calls: 3 }),
  }));
  r.ev('renderRunNote()');
  assert.ok(r.runNoteEl.textContent.includes('후보 묶음 1개 중 0개 리랭킹'), r.runNoteEl.textContent);
  assert.ok(r.runNoteEl.textContent.includes('모델 호출 3회'), '재시도 3회가 묶음 기록 1과 다르게 보인다');
  assert.equal(r.runNoteEl.classList.contains('warn'), true);

  r.ev('convo = ' + JSON.stringify({
    notice: null,
    explain: runExplain({ reorder_stage: 'policy_skipped', reorder_stage_label: '리랭킹 생략(정책)',
                          rerank_calls: 0, rerank_calls_applied: 0, rerank_model_calls: 0 }),
  }));
  r.ev('renderRunNote()');
  assert.ok(!r.runNoteEl.textContent.includes('모델 호출'), '부르지 않은 요청에 호출 횟수를 적지 않는다');
  assert.equal(r.runNoteEl.classList.contains('warn'), false);
});
