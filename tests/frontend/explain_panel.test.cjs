// node --test tests/frontend/explain_panel.test.cjs   (pytest도 tests/test_frontend_node.py로 함께 돌린다)
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
  'explainMix', 'explainLyric', 'explainEvidence', 'explainPanel', 'renderRunNote',
  'applyResponse', 'isBirthYearQuestion',
];
const PROGRAM = [...DECLARATIONS.map(declaration), ...FUNCTIONS.map(functionSource)].join('\n');

/* 아주 작은 DOM. 텍스트를 모아 "무엇이 보이는가"만 묻는다 */
function node(tag) {
  const classes = new Set();
  return {
    tag, className: '', textContent: '', children: [], hidden: false, attrs: {},
    classList: {
      add(c) { classes.add(c); },
      toggle(c, on) { if (on) classes.add(c); else classes.delete(c); },
      contains(c) { return classes.has(c); },
    },
    append(...items) { this.children.push(...items); },
    setAttribute(k, v) { this.attrs[k] = v; },
  };
}
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
    explainOpen: new Set(), markHits() { return 0; }, renderPanel() {}, resultList: { scrollTop: 0 },
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
});
