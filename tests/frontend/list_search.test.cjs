// node --test tests/frontend/list_search.test.cjs   (pytest도 tests/frontend/test_frontend_node.py로 함께 돌린다)
//
// 목록 검색창이 고른 기준(제목/가수)으로만 곡을 좁히는지 **실제 함수로** 확인한다. DOM만 대체한다.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../src/frontend/static/map.js'), 'utf8');

function functionSource(name) {
  const found = new RegExp('^function ' + name + '\\(', 'm').exec(source);
  assert(found, `map.js에 최상위 함수 ${name}이 없다`);
  const end = source.indexOf('\n}', found.index);
  return source.slice(found.index, end + 2);
}

const PROGRAM = ['normalizeListSearch', 'matchesListSearch', 'sortedSongs'].map(functionSource).join('\n');

const SONGS = [
  { id: 'a', ft: '봄날', a: '방탄소년단', g: '댄스', d: '2017-02-13' },
  { id: 'b', ft: '방탄 이야기', a: '아이유', g: '발라드', d: '2015-01-01' },
  { id: 'c', ft: '사랑 했지만', a: '김광석', g: '발라드', d: '1991-01-01' },
  // 자료에 실제로 있는 자모 분리형(NFD) 제목
  { id: 'd', ft: '겨울 편지'.normalize('NFD'), a: 'Busker Busker', g: '록', d: '2012-03-29' },
];

/* 입력값을 넣고 sortedSongs가 내는 곡 ID */
function search(field, value, { genre = '', heardIds = [] } = {}) {
  const ctx = {
    genreSelect: { value: genre },
    listSearchField: { value: field },
    listSearchInput: { value },
    heardOnly: { checked: heardIds.length > 0 },
    heard: new Set(heardIds),
    songsById: new Map(SONGS.map((s) => [s.id, s])),
    listSort: 'title',
    listDesc: false,
    collator: new Intl.Collator('ko', { numeric: true, sensitivity: 'base' }),
  };
  vm.createContext(ctx);
  vm.runInContext(PROGRAM, ctx);
  return [...ctx.sortedSongs()].map((s) => s.id).sort();
}

test('제목 기준은 제목에서만 찾는다', () => {
  // '방탄'은 b의 제목이고 a의 가수 이름이다
  assert.deepEqual(search('title', '방탄'), ['b']);
});

test('가수 기준은 가수에서만 찾는다', () => {
  assert.deepEqual(search('artist', '방탄'), ['a']);
});

test('검색어가 비었거나 공백뿐이면 전부 나온다', () => {
  assert.deepEqual(search('title', ''), ['a', 'b', 'c', 'd']);
  assert.deepEqual(search('artist', '   '), ['a', 'b', 'c', 'd']);
});

test('띄어쓰기와 대소문자를 가리지 않는다', () => {
  assert.deepEqual(search('title', '사랑했지만'), ['c']);
  assert.deepEqual(search('title', '사랑 했 지만'), ['c']);
  assert.deepEqual(search('artist', 'buskerbusker'), ['d']);
});

test('자모 분리형(NFD) 제목도 찾는다', () => {
  assert.deepEqual(search('title', '겨울'), ['d']);
});

test('없는 검색어는 0곡이다', () => {
  assert.deepEqual(search('title', '김광석'), []);
});

test('장르·들은 곡 필터와 함께 걸린다', () => {
  assert.deepEqual(search('artist', '아이유', { genre: '댄스' }), []);
  assert.deepEqual(search('title', '봄날', { genre: '댄스' }), ['a']);
  assert.deepEqual(search('title', '날', { heardIds: ['b'] }), []);
  assert.deepEqual(search('title', '이야기', { heardIds: ['b'] }), ['b']);
});
