// node --test tests/frontend/model_picker.test.cjs   (pytest의 tests/test_frontend_node.py가 함께 돌린다)
//
// 검색 모델 선택창(map.js의 loadSearchModels)이 **사용 가능한 모델만** 고르는지 본다. 서버가 "사용 불가"라고 한
// 기본값을 그대로 고르면 모든 검색이 503이다. DOM·fetch·localStorage만 대체한다.
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

const PROGRAM = [declaration('SEARCH_PATH'), declaration('MODEL_KEY'), functionSource('loadSearchModels')].join('\n');

function model(provider, available) {
  return { provider, label: provider, model: `${provider}-model`, available };
}

/* 선택창 하나. info는 GET /search/models 응답, saved는 localStorage에 남아 있던 선택 */
async function load(info, saved = '') {
  const options = [];
  const select = { value: '', replaceChildren() { options.length = 0; }, append(o) { options.push(o); } };
  const storage = new Map(saved ? [['vf.modelProvider', saved]] : []);
  const ctx = {
    modelProvider: select,
    modelHint: { textContent: '' },
    document: { createElement: () => ({}) },
    localStorage: { getItem: key => (storage.has(key) ? storage.get(key) : null), setItem: (k, v) => storage.set(k, v) },
    AbortSignal: { timeout: () => undefined },
    fetch: async () => ({ ok: true, json: async () => info }),
  };
  vm.createContext(ctx);
  vm.runInContext(PROGRAM, ctx);
  await vm.runInContext('loadSearchModels()', ctx);
  return { select, options, hint: ctx.modelHint.textContent };
}

test('저장한 선택이 사용 가능하면 그것을 고른다', async () => {
  const s = await load({ default: 'claude', models: [model('claude', true), model('gemini', true)] }, 'gemini');
  assert.equal(s.select.value, 'gemini');
  assert.deepEqual(s.options.map(o => o.disabled), [false, false]);
});

test('저장한 선택이 사용 불가면 서버 기본값으로 간다', async () => {
  const s = await load({ default: 'claude', models: [model('claude', true), model('gemini', false)] }, 'gemini');
  assert.equal(s.select.value, 'claude');
});

test('서버 기본값이 사용 불가면 첫 사용 가능 모델을 고른다 — 503이 날 선택을 자동으로 하지 않는다', async () => {
  const s = await load({ default: 'claude', models: [model('claude', false), model('gemini', true)] });
  assert.equal(s.select.value, 'gemini');
  assert.match(s.hint, /다음 새 검색부터/);
});

test('사용 가능한 모델이 없으면 고르지 않고 키 설정을 안내한다', async () => {
  const s = await load({ default: 'claude', models: [model('claude', false), model('gemini', false)] }, 'claude');
  assert.equal(s.select.value, '');
  assert.match(s.hint, /API 키/);
});
