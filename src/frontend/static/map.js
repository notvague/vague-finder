/* Vague Finder  노래 맵 */
"use strict";

/*
 * 재생은 YouTube 임베드에 위임하고, 커버는 멜론 CDN 원본을 참조한다.
 * 음원·이미지를 자체 호스팅하지 않으므로 저작물을 재배포하지 않는다.
 */

const GENRE_COLORS = {
  "발라드": "#3f7fc1",
  "댄스": "#e0529c",
  "랩/힙합": "#9c4fd4",
  "R&B/Soul": "#d98a1f",
  "록/메탈": "#c14b3f",
  "국내드라마": "#55a047",
  "인디음악": "#2fa3a0",
  "포크/블루스": "#8a7a2f",
  "일렉트로니카": "#7f6fe0",
  "J-POP": "#3aa5d9",
  "성인가요/트로트": "#a0522d",
  "재즈": "#2e8b57",
  "국내뮤지컬": "#a3378f",
};
const FALLBACK_COLOR = "#777";

const canvas = document.getElementById("canvas");
const legend = document.getElementById("legend");
const axisLeft = document.getElementById("axisLeft");
const axisRight = document.getElementById("axisRight");
const axisTop = document.getElementById("axisTop");
const axisBottom = document.getElementById("axisBottom");
const axisLineX = document.getElementById("axisLineX");
const axisLineY = document.getElementById("axisLineY");
const card = document.getElementById("card");
const cardCover = document.getElementById("cardCover");
const cardTitle = document.getElementById("cardTitle");
const cardArtist = document.getElementById("cardArtist");
const cardPlayer = document.getElementById("cardPlayer");
const playBtn = document.getElementById("playBtn");
const listView = document.getElementById("listview");
const listBody = document.getElementById("listBody");
const listCount = document.getElementById("listCount");
const genreSelect = document.getElementById("genreSelect");
const artistInput = document.getElementById("artistInput");
const artistList = document.getElementById("artistList");
const artistClear = document.getElementById("artistClear");
const heardOnly = document.getElementById("heardOnly");
const searchForm = document.getElementById("searchForm");
const searchInput = document.getElementById("searchInput");
const clearBtn = document.getElementById("clearBtn");
const statusBar = document.getElementById("status");
const rail = document.getElementById("rail");
const railToggle = document.getElementById("railToggle");
const topbar = document.querySelector(".topbar");
const root = document.documentElement;

let songsById = new Map();
let playingEl = null;
let playingId = null;
let cardId = null;
let axes = null;
let currentMap = "sound";
/*
 * 파이프라인은 1800px 폭 기준으로 좌표를 만든다. 그 값을 그대로 쓰면
 * 넓은 화면엔 오른쪽 여백이, 좁은 화면엔 가로 스크롤이 생긴다.
 * 그래서 좌표 공간은 그대로 두고, 그릴 때만 화면 폭에 맞춰 배율을 건다.
 */
let baseCanvas = { width: 1800, height: 3016 };
/*
 * 이보다 좁은 화면에서는 축소하지 않고 원래 1800px 크기로 되돌린다.
 * 1100px 같은 중간값으로 줄이면 글자 크기는 그대로인데 좌표만 좁아져
 * 제목들이 서로 겹쳐 읽을 수 없게 된다. 그냥 가로 스크롤로 보는 게 낫다.
 */
const RESPONSIVE_MIN_WIDTH = 1400;

/*
 * 왼쪽 레일이 지도에서 가져간 폭. 좁은 화면에서는 레일이 위로 눕고,
 * 접으면 화면 밖으로 밀려나므로 두 경우 모두 0이 된다.
 * 이 미디어 질의는 style.css의 @media (max-width: 900px)와 같아야 한다.
 */
let railW = 0;
const stackedRail = window.matchMedia("(max-width: 900px)");

/*
 * 청취 기록 — 새로고침해도 남아야 기록으로서 의미가 있으므로 localStorage에 둔다.
 * 재생 중 표시(♫, played)와 별개다: played는 한 곡뿐이고 멈추면 사라진다.
 */
const HEARD_KEY = "vf.heard";
const heard = new Set(loadHeard());

function loadHeard() {
  try {
    const raw = localStorage.getItem(HEARD_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return []; // 사생활 보호 모드 등에서 접근이 막히면 이번 세션만 기록한다
  }
}

function markHeard(id) {
  if (heard.has(id)) return;
  heard.add(id);
  try {
    localStorage.setItem(HEARD_KEY, JSON.stringify([...heard]));
  } catch {
    /* 저장 실패해도 화면 표시는 유지된다 */
  }
}

function clearHeard() {
  heard.clear();
  try {
    localStorage.removeItem(HEARD_KEY);
  } catch {
    /* 무시 */
  }
  for (const s of songsById.values()) s.el.removeAttribute("heard");
  renderList();
}
window.clearHeard = clearHeard; // 콘솔에서 기록 초기화용

/*
 * 출생 연도 — "중학교 때 듣던" 같은 생애 단계 질의를 연도로 바꾸는 기준점. 서버가 birth_year 슬롯으로
 * 한 번 묻고, 답은 여기 저장해 다음 검색부터 요청의 birth_year로 보낸다(같은 사람에게 다시 묻지 않는다).
 * 서버는 저장하지 않는다. 지우려면 콘솔에서 clearBirthYear().
 */
const BIRTH_YEAR_KEY = "vf.birthYear";
let sessionBirthYear = null; // 저장소를 못 쓰는 브라우저(사생활 창 등)에서도 이 탭 안에서는 다시 묻지 않게

function loadBirthYear() {
  try {
    const raw = localStorage.getItem(BIRTH_YEAR_KEY);
    const year = raw ? Number(raw) : NaN;
    if (Number.isInteger(year) && year >= 1900 && year <= 2100) return year;
  } catch {
    /* 저장소를 못 읽으면 이 탭의 값으로 */
  }
  return sessionBirthYear;
}

function saveBirthYear(year) {
  sessionBirthYear = year;
  try {
    localStorage.setItem(BIRTH_YEAR_KEY, String(year));
  } catch {
    /* 저장 못 해도 이 탭 안에서는 sessionBirthYear로 보낸다 */
  }
}

function clearBirthYear() {
  sessionBirthYear = null;
  try {
    localStorage.removeItem(BIRTH_YEAR_KEY);
  } catch {
    /* ignore */
  }
}
window.clearBirthYear = clearBirthYear;

/* '1996~2000년생' → 1998(밴드 중앙). 서버 parse_birth_year와 같은 규칙 */
function parseBirthYear(value) {
  const years = String(value || "").match(/(?:19|20)\d{2}/g);
  if (!years) return null;
  const nums = years.slice(0, 2).map(Number);
  return Math.round(nums.reduce((a, b) => a + b, 0) / nums.length);
}

/*
 * 맵은 두 종류다. 두 임베딩의 이웃 집합이 3.3%밖에 겹치지 않아
 * (소리가 닮은 곡과 정서가 닮은 곡은 서로 다른 집합) 한 장에 합치지 않고
 * 탭으로 나눈다. 곡 목록은 그대로 두고 좌표만 바꿔 끼운다.
 */
const MAP_COORDS = {
  sound: (s) => [s.x, s.y],
  mood: (s) => [s.mx, s.my],
};

function applyAxisLabels() {
  const a = (axes && axes[currentMap]) || { x: ["", ""], y: ["", ""] };
  axisLeft.textContent = a.x[0] || "";
  axisRight.textContent = a.x[1] || "";
  axisTop.textContent = a.y[0] || "";
  axisBottom.textContent = a.y[1] || "";
  // 의미 없는 축에는 라벨을 붙이지 않는다 (sound 맵의 가로축)
  for (const el of [axisLeft, axisRight, axisTop, axisBottom]) {
    el.hidden = !el.textContent;
  }
  // 어떤 선을 그릴지는 맵 데이터가 정한다 (axes[map].lines: ["x","y"])
  const lines = a.lines || [];
  axisLineX.hidden = !lines.includes("x");
  axisLineY.hidden = !lines.includes("y");
}

function canvasScale() {
  /*
   * innerWidth에는 세로 스크롤바가 포함된다. 곡을 전부 그리면 반드시
   * 세로 스크롤이 생기므로, 스크롤바 폭을 미리 빼야 가로 스크롤이 안 생긴다.
   * (clientWidth로 재면 스크롤바가 아직 없는 초기 시점에 과대 측정된다.)
   * 왼쪽 레일이 가져간 폭도 뺀다 — 지도는 레일 오른쪽에만 그려진다.
   */
  const sb = window.innerWidth - document.documentElement.clientWidth;
  const avail = window.innerWidth - (sb || scrollbarWidth()) - railW;
  // 1400px 미만은 축소하지 않고 원래 좌표 그대로 쓴다 (줄이면 제목이 겹친다)
  if (avail < RESPONSIVE_MIN_WIDTH) return 1;
  // 1400px 이상은 화면 폭에 맞춘다 — 넓으면 늘리고, 1800px보다 좁으면 줄인다
  return avail / baseCanvas.width;
}

let sbCache = null;
function scrollbarWidth() {
  if (sbCache !== null) return sbCache;
  const probe = document.createElement("div");
  probe.style.cssText =
    "position:absolute;top:-9999px;width:100px;height:100px;overflow:scroll";
  document.body.appendChild(probe);
  sbCache = probe.offsetWidth - probe.clientWidth;
  probe.remove();
  return sbCache;
}

/*
 * 레일이 지도에서 얼마를 가져갔는지 CSS와 JS가 같은 값을 보게 맞춘다.
 * 넓은 화면은 왼쪽에 세로로 서므로 폭(--rail-w)을, 좁은 화면은 위에 눕으므로
 * 높이(--rail-h)를 넘긴다. 접으면 둘 다 0이다.
 * offsetWidth/Height는 레이아웃 값이라 접을 때 건 transform에 흔들리지 않는다.
 */
function syncHeaderHeight() {
  // 헤더 높이는 폭과 서체에 따라 달라진다. 레일·지도·목록이 그만큼 아래에서 시작한다
  const h = Math.ceil(topbar.getBoundingClientRect().height);
  root.style.setProperty("--header-h", h + "px");
}

function syncRailMetrics() {
  const stacked = stackedRail.matches;
  const closed = document.body.classList.contains("railclosed");
  railW = stacked || closed ? 0 : rail.offsetWidth + 1; // +1 = 오른쪽 테두리
  root.style.setProperty("--rail-w", railW + "px");
  root.style.setProperty("--rail-h", (stacked ? rail.offsetHeight : 0) + "px");
}

function resizeCanvas() {
  // 레일이 헤더 아래에서 시작하므로 헤더를 먼저 잰다
  syncHeaderHeight();
  syncRailMetrics();
  const k = canvasScale();
  const w = Math.round(baseCanvas.width * k);
  canvas.style.width = w + "px";
  canvas.style.height = Math.round(baseCanvas.height * k) + "px";
  // footer가 캔버스와 같은 폭을 채워야 가로 스크롤 시 잘리지 않는다
  document.documentElement.style.setProperty("--canvas-width", w + "px");
  return k;
}

function placeSongs() {
  const pick = MAP_COORDS[currentMap];
  const k = resizeCanvas();
  for (const song of songsById.values()) {
    const [x, y] = pick(song);
    song.el.style.left = (x * k).toFixed(1) + "px";
    song.el.style.top = (y * k).toFixed(1) + "px";
  }
}

function switchMap(name) {
  if (!MAP_COORDS[name] || name === currentMap) return;
  currentMap = name;
  for (const el of document.querySelectorAll(".mode[data-map]")) {
    el.classList.toggle("current", el.dataset.map === name);
  }
  closeCard();
  stopPlayback();
  placeSongs();
  applyAxisLabels();
  window.scrollTo({ top: 0, left: 0 });
}

async function init() {
  // 맵을 다시 빌드해도 브라우저가 옛 좌표를 캐시하지 않도록
  const res = await fetch("/static/map_data.json", { cache: "no-store" });
  const data = await res.json();
  axes = data.axes || null;
  baseCanvas = data.canvas;

  const frag = document.createDocumentFragment();
  for (const s of data.songs) {
    const el = document.createElement("div");
    el.className = "song";
    el.textContent = s.t;
    el.style.color = GENRE_COLORS[s.g] || FALLBACK_COLOR;
    el.dataset.id = s.id;
    if (heard.has(s.id)) el.setAttribute("heard", "");
    frag.appendChild(el);
    songsById.set(s.id, { ...s, el });
  }
  canvas.appendChild(frag);
  placeSongs();
  /*
   * 곡을 넣으면 세로 스크롤바가 생기면서 가용 폭이 줄어든다.
   * 스크롤바가 실제로 그려진 뒤의 폭으로 한 번 더 잡아야 가로 스크롤이 안 생긴다.
   */
  requestAnimationFrame(() => requestAnimationFrame(placeSongs));
  applyAxisLabels();
  fillGenreOptions();
  buildLegend();
}

/*
 * 범례는 지도를 볼 때, 검색 결과가 없을 때만 띄운다.
 * 목록 보기에는 표에 장르 열이 있어 필요 없고, 검색 결과가 있으면
 * 레일에서 같은 자리를 놓고 다투므로 결과에 양보한다.
 */
function updateLegendVisibility() {
  legend.hidden = !listView.hidden || !!convo;
}

function buildLegend() {
  // 색 정의(GENRE_COLORS)에서 바로 만들어 지도와 항상 일치하게 한다
  const counts = new Map();
  for (const s of songsById.values()) {
    counts.set(s.g, (counts.get(s.g) || 0) + 1);
  }
  const frag = document.createDocumentFragment();
  const head = document.createElement("div");
  head.className = "legendhead";
  head.textContent = `장르 · ${songsById.size}곡`;
  frag.appendChild(head);
  for (const [genre, n] of [...counts].sort((a, b) => b[1] - a[1])) {
    const row = document.createElement("div");
    row.className = "legendrow";
    const chip = document.createElement("span");
    chip.className = "legendchip";
    chip.style.background = GENRE_COLORS[genre] || FALLBACK_COLOR;
    const name = document.createElement("span");
    name.className = "legendname";
    name.textContent = genre;
    const num = document.createElement("span");
    num.className = "legendnum";
    num.textContent = n;
    row.append(chip, name, num);
    frag.appendChild(row);
  }
  legend.replaceChildren(frag);
}

function fillGenreOptions() {
  const counts = new Map();
  const artists = new Map();
  for (const s of songsById.values()) {
    counts.set(s.g, (counts.get(s.g) || 0) + 1);
    artists.set(s.a, (artists.get(s.a) || 0) + 1);
  }
  for (const [genre, n] of [...counts].sort((a, b) => b[1] - a[1])) {
    const o = document.createElement("option");
    o.value = genre;
    o.textContent = `${genre} (${n})`;
    genreSelect.appendChild(o);
  }
  // 가수가 천 명이 넘어 곡 수 많은 순으로 제안한다
  const frag = document.createDocumentFragment();
  for (const [name, n] of [...artists].sort(
    (a, b) => b[1] - a[1] || collator.compare(a[0], b[0])
  )) {
    const o = document.createElement("option");
    o.value = name;
    o.label = `${n}곡`;
    frag.appendChild(o);
  }
  artistList.appendChild(frag);
}

for (const el of document.querySelectorAll(".mode[data-map]")) {
  el.addEventListener("click", () => {
    showMapView();
    switchMap(el.dataset.map);
  });
}

/* ---------- 목록 보기 ---------- */
const listTab = document.querySelector('.mode[data-view="list"]');

let listSort = "title";
// 각 정렬의 기본 방향: 제목은 ㄱ→ㅎ, 발매일은 최신 먼저
const SORT_DEFAULT_DESC = { title: false, date: true };
let listDesc = SORT_DEFAULT_DESC[listSort];
// 한글 정렬은 기본 문자열 비교로는 자모 순서가 어긋나므로 ko 로케일을 쓴다
const collator = new Intl.Collator("ko", { numeric: true, sensitivity: "base" });

function sortedSongs() {
  const genre = genreSelect.value;
  const artist = artistInput.value.trim().toLowerCase();
  const rows = [...songsById.values()].filter(
    (s) =>
      (!genre || s.g === genre) &&
      (!artist || s.a.toLowerCase().includes(artist)) &&
      (!heardOnly.checked || heard.has(s.id))
  );
  const dir = listDesc ? -1 : 1;
  rows.sort((a, b) => {
    if (listSort === "date") {
      // 같은 날짜끼리는 방향과 무관하게 항상 가나다순으로 묶어 보여준다
      const d = (a.d || "").localeCompare(b.d || "");
      return d ? d * dir : collator.compare(a.ft, b.ft);
    }
    return collator.compare(a.ft, b.ft) * dir;
  });
  return rows;
}

function updateSortButtons() {
  for (const btn of document.querySelectorAll(".sort")) {
    const active = btn.dataset.sort === listSort;
    btn.classList.toggle("current", active);
    const arrow = btn.querySelector(".arrow");
    if (arrow) arrow.textContent = active ? (listDesc ? " ↓" : " ↑") : "";
    btn.title = active
      ? "다시 누르면 정렬 방향이 바뀝니다"
      : "이 기준으로 정렬";
  }
}

function renderList() {
  const rows = sortedSongs();
  const frag = document.createDocumentFragment();
  rows.forEach((s, i) => {
    const tr = document.createElement("tr");
    tr.dataset.id = s.id;
    if (heard.has(s.id)) tr.classList.add("heardrow");
    const cells = [
      String(i + 1),
      s.ft,
      s.a,
      s.g,
      (s.d || "").replace(/-/g, "."),
    ];
    cells.forEach((text, col) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (col === 0) td.className = "numcol";
      if (col === 3) td.style.color = GENRE_COLORS[s.g] || FALLBACK_COLOR;
      if (col === 4) td.className = "datecol";
      tr.appendChild(td);
    });
    frag.appendChild(tr);
  });
  listBody.replaceChildren(frag);
  listCount.textContent = `${rows.length}곡`;
}

function showListView() {
  listView.hidden = false;
  canvas.hidden = true;
  document.body.classList.add("listmode");
  updateLegendVisibility();
  closeCard();
  for (const el of document.querySelectorAll(".mode")) el.classList.remove("current");
  listTab.classList.add("current");
  updateSortButtons();
  renderList();
}

function showMapView() {
  if (listView.hidden) return;
  listView.hidden = true;
  canvas.hidden = false;
  document.body.classList.remove("listmode");
  updateLegendVisibility();
  listTab.classList.remove("current");
  /*
   * 지금 보고 있는 맵 탭에 표시를 되돌린다. switchMap은 같은 맵이면 일찍
   * 빠져나가므로(sound → list → sound) 여기서 해 주지 않으면 아무 탭에도
   * 표시가 없는 상태가 된다.
   */
  for (const el of document.querySelectorAll(".mode[data-map]")) {
    el.classList.toggle("current", el.dataset.map === currentMap);
  }
}

listTab.addEventListener("click", showListView);
genreSelect.addEventListener("change", renderList);
heardOnly.addEventListener("change", renderList);

for (const btn of document.querySelectorAll(".sort")) {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sort;
    // 이미 활성인 버튼을 누르면 방향 토글, 다른 버튼이면 그 정렬의 기본 방향
    listDesc = key === listSort ? !listDesc : SORT_DEFAULT_DESC[key];
    listSort = key;
    updateSortButtons();
    renderList();
  });
}

let artistDebounce = null;
artistInput.addEventListener("input", () => {
  artistClear.hidden = !artistInput.value;
  // 타이핑마다 수천 행을 다시 그리면 버벅이므로 잠깐 모아서 처리한다
  clearTimeout(artistDebounce);
  artistDebounce = setTimeout(renderList, 120);
});
artistClear.addEventListener("click", () => {
  artistInput.value = "";
  artistClear.hidden = true;
  renderList();
  artistInput.focus();
});

// 목록에서 곡을 클릭하면 해당 곡 위치로 지도를 열어준다
listBody.addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (!tr) return;
  // 이 클릭이 document 핸들러까지 가면 방금 연 카드가 곧바로 닫힌다
  e.stopPropagation();
  showMapView();
  openCard(tr.dataset.id);
  const song = songsById.get(tr.dataset.id);
  if (song) song.el.scrollIntoView({ block: "center", inline: "center" });
});

/* ---------- 재생 (YouTube 임베드) ---------- */
function stopPlayback() {
  cardPlayer.innerHTML = "";
  cardPlayer.hidden = true;
  if (playingEl) {
    playingEl.classList.remove("playing");
    // 음표는 '재생 중' 표시이므로 멈추면 지운다 (안 지우면 계속 남는다)
    playingEl.removeAttribute("played");
  }
  playingEl = null;
  playingId = null;
  playBtn.innerHTML = "&#9654;";
}

function togglePlay(id) {
  if (playingId === id) {
    stopPlayback();
    return;
  }
  const song = songsById.get(id);
  if (!song) return;
  if (!song.yt) {
    showStatus("이 곡은 재생할 수 있는 영상이 없습니다.");
    return;
  }
  if (playingEl) {
    playingEl.classList.remove("playing");
    playingEl.removeAttribute("played");
  }

  const frame = document.createElement("iframe");
  frame.src = `https://www.youtube-nocookie.com/embed/${song.yt}?autoplay=1`;
  frame.title = `${song.ft} - ${song.a}`;
  frame.allow = "autoplay; encrypted-media";
  frame.referrerPolicy = "strict-origin-when-cross-origin";
  frame.setAttribute("allowfullscreen", "");
  cardPlayer.replaceChildren(frame);
  cardPlayer.hidden = false;

  song.el.setAttribute("played", "");
  song.el.setAttribute("heard", "");
  markHeard(id);
  song.el.classList.add("playing");
  playingEl = song.el;
  playingId = id;
  playBtn.innerHTML = "&#10074;&#10074;";
}

/* ---------- 곡 카드 (클릭 시 곡 텍스트 옆에 고정) ---------- */
function openCard(id) {
  const song = songsById.get(id);
  if (!song) return;
  cardId = id;
  cardTitle.textContent = song.ft;
  cardArtist.textContent = `${song.a} · ${song.g}`;
  cardCover.src = song.c;
  playBtn.hidden = !song.yt;
  // 다른 곡 카드를 열면 이전 곡 영상은 닫는다 (카드 안에 플레이어가 있으므로)
  if (playingId !== id) stopPlayback();
  playBtn.innerHTML = playingId === id ? "&#10074;&#10074;" : "&#9654;";
  card.hidden = false;
  // 곡 텍스트의 오른쪽 아래에 고정 (맵과 함께 스크롤됨)
  const pad = 14;
  const cw = card.offsetWidth;
  const ch = card.offsetHeight;
  const k = canvasScale(); // 곡 위치가 배율 적용된 값이라 카드도 같은 배율로
  const [bx, by] = MAP_COORDS[currentMap](song); // 탭마다 좌표가 다르다
  const sx = bx * k;
  const sy = by * k;
  let x = sx + pad;
  let y = sy + pad;
  if (x + cw > canvas.clientWidth - 4) x = sx - cw - pad; // 오른쪽 끝이면 왼쪽에
  if (y + ch > canvas.clientHeight - 4) y = sy - ch - pad; // 아래 끝이면 위에
  card.style.left = Math.max(4, x) + "px";
  card.style.top = Math.max(4, y) + "px";
}

function closeCard() {
  card.hidden = true;
  cardId = null;
}

canvas.addEventListener("click", (e) => {
  const el = e.target.closest(".song");
  if (!el) return;
  if (cardId === el.dataset.id && !card.hidden) {
    closeCard();
    return;
  }
  openCard(el.dataset.id);
});

playBtn.addEventListener("click", () => {
  if (cardId) togglePlay(cardId);
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".song") && !e.target.closest(".card")) closeCard();
});

/* ---------- 검색 ---------- */
/*
 * 검색창 바로 아래 한 줄. tone="warn"은 서버에 닿지 못해 결과를 믿을 수 없는
 * 경우에만 쓴다 — "검색 중…" 같은 진행 문구까지 경고색이면 구분이 사라진다.
 */
function showStatus(msg, tone) {
  statusBar.textContent = msg;
  statusBar.hidden = !msg;
  statusBar.classList.toggle("warn", tone === "warn");
}

function clearSearch() {
  canvas.classList.remove("searching");
  for (const { el } of songsById.values()) {
    el.classList.remove("hit", "top");
    delete el.dataset.rank;
  }
  searchInput.value = "";
  clearBtn.hidden = true;
  showStatus("");
  searchSeq += 1; // 진행 중인 요청의 응답은 버린다
  convo = null;
  panelMode = "idle";
  renderPanel();
}

function markHits(ids) {
  canvas.classList.add("searching");
  for (const { el } of songsById.values()) {
    el.classList.remove("hit", "top");
    delete el.dataset.rank;
  }
  let first = null;
  let rank = 0;
  for (const id of ids) {
    const song = songsById.get(String(id));
    if (!song) continue;
    rank += 1;
    song.el.classList.add("hit");
    song.el.dataset.rank = String(rank);
    if (!first) {
      first = song.el;
      first.classList.add("top");
    }
  }
  if (first) first.scrollIntoView({ block: "center", inline: "center" });
  return rank;
}

function localSearch(q) {
  const needle = q.toLowerCase();
  const ids = [];
  for (const s of songsById.values()) {
    if (s.ft.toLowerCase().includes(needle) || s.a.toLowerCase().includes(needle)) {
      ids.push(s.id);
    }
  }
  return ids;
}

/*
 * 제한 시간을 둔다. 없으면 서버가 응답하지 않을 때 화면이 "검색 중…"에서 끝나지
 * 않는다 — 2026-09-24 실패 경로 확인에서 15초를 더 기다려도 벗어나지 못했고,
 * 발표 중에는 새로고침 말고 빠져나갈 길이 없다는 뜻이었다.
 *
 * 30초인 이유: 예열 직후 첫 검색이 5.6초, 예열 전 최악이 17.2초였다
 * (experiments/latency). 정상 검색은 자르지 않으면서 멈춘 화면은 끝낸다.
 * 서버 쪽 질의 분석에도 따로 20초 예산이 있다(QUERY_ANALYSIS_TIMEOUT_SECONDS).
 */
const SEARCH_TIMEOUT_MS = 30000;
const SEARCH_PATH = "/api/v1/search";
/*
 * 질문 조회는 검색이 아니다 — 후보 20곡 안팎의 메타데이터만 읽는다. 검색과 같은 30초를 주면 서버가
 * 멈췄을 때 "이 중에는 없어요"가 30초 동안 먹통이 된다. 이 시간이 지나면 질문 없이 거절만 진행한다.
 */
const CLARIFY_PATH = "/api/v1/search/clarify";
const CLARIFY_TIMEOUT_MS = 8000;

async function requestSearch(body, path = SEARCH_PATH, timeoutMs = SEARCH_TIMEOUT_MS) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const err = new Error(`search ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/*
 * 왜 의미 검색을 못 했는가. **원인마다 다른 말을 해야 한다.**
 *
 * 예전에는 서버가 500을 내도 문자열 검색이 0건이면 "일치하는 곡이 없습니다"만
 * 떴다. 서버가 죽은 것과 정말 없는 것이 같은 문구가 되어, 발표자도 관객도
 * 데이터에 그 곡이 없다고 읽는다.
 */
function searchFailureText(err) {
  if (err && err.name === "TimeoutError") {
    return `검색이 ${SEARCH_TIMEOUT_MS / 1000}초 안에 끝나지 않았습니다.`;
  }
  if (err && err.status) return `검색 서버가 오류로 응답했습니다(${err.status}).`;
  return "의미 검색 서버에 연결하지 못했습니다.";
}

/* ---------- 재질문 (2턴) ----------
 *
 * 서버는 상태를 저장하지 않는다. 응답의 analysis·asked_slots·rejected_ids·turn을
 * 여기 보관했다가 다음 요청에 그대로 되돌려준다.
 *
 * 흐름:
 *   검색 → Top-10 (출생 연도 질문만 즉시 표시)
 *   "이 중에는 없어요" → 현재 후보로 질문을 조회하고(/search/clarify), 없으면 거절만 해서 다시 검색
 *   답 선택 / "잘 모르겠어요" → 보여준 10곡을 거절 + 답변을 실어 다시 검색
 *
 * 질문 조회는 **질문만** 가져온다 — 턴·거절 목록·결과를 바꾸지 않는다. 그래서 질문을 닫아도 잃는 것이
 * 없고, 다시 누르면 받아 둔 질문을 그대로 연다(결과가 바뀌면 applyResponse가 버린다). 조회가 실패하거나
 * 제한 시간을 넘기면 안내를 남기고 질문 없이 거절만 진행한다 — 거절에는 질문이 필요 없다.
 *
 * 예외 — 출생 연도(birth_year) 질문은 곡이 아니라 사용자에 대한 것이라 거절을 기다리지 않고 **결과와 함께 바로** 묻는다.
 * 답하면 저장하고 같은 질의를 **기존 분석 + birth_year로** 다시 검색한다(재분석 없음·거절 없음·턴 그대로 — 첫 결과부터
 * 시기 창이 들어가야 한다). 실패하면 앞의 결과를 그대로 둔다. 답은 저장돼 있으니 다음 검색부터 반영된다.
 * 연령대는 이 재검색이 돌려준 후보(candidateIds)에 반영된다 — 그 뒤의 질문 조회는 그 후보를 보낼 뿐 연도를 다시 보내지 않는다.
 * "잘 모르겠어요"·닫기는 이 검색에서는 다시 묻지 않고, 그 뒤 "이 중에는 없어요"를 누르면 데이터 질문을 조회한다.
 *
 * 질문을 조회할지는 서버가 정한다(clarify_deferred — 한도가 남았고 보여준 곡 말고 후보가 있을 때). 화면은 거절
 * 한도만 따로 확인한다 — 질문 없이 거절만 하는 경로가 있어서다.
 */
const TOP_K = 10;
// 서버 스키마 MAX_REJECTED_IDS와 같아야 한다. 넘기면 요청이 422로 막힌다.
const MAX_REJECTED = 20;
// 서버 스키마의 previous_candidate_ids max_length
const MAX_PREVIOUS_CANDIDATES = 30;

const resultBlock = document.getElementById("resultBlock");
const runNote = document.getElementById("runNote");
const resultSub = document.getElementById("resultSub");
const resultList = document.getElementById("resultList");
const clarifyBox = document.getElementById("clarifyBox");

let convo = null;       // 진행 중인 재질문 대화. 새 검색마다 초기화
let panelMode = "idle"; // idle | asking | loading(다시 찾는 중) | question_loading(질문 조회 중)
let searchSeq = 0;      // 늦게 도착한 옛 응답이 새 검색 결과를 덮지 않게

/* 요청이 나가 있는 동안에는 버튼·답·거절을 모두 막는다 — 같은 거절이 두 번 나가지 않게 */
function isBusy() {
  return panelMode === "loading" || panelMode === "question_loading";
}

function startConvo(query, data) {
  convo = { query, answers: [], birthYearSkipped: false };
  // 좁은 화면은 접힌 상태에서도 검색줄이 보인다. 거기서 검색했으면 펼쳐 준다
  setRailClosed(false);
  applyResponse(data);
}

function applyResponse(data) {
  convo.analysis = data.analysis || null;
  convo.askedSlots = data.asked_slots || [];
  convo.rejectedIds = data.rejected_ids || [];
  convo.candidateIds = data.candidate_ids || [];
  convo.turn = data.turn || 1;
  convo.results = data.results || [];
  // 결과·후보가 바뀌었다 — 앞 결과로 받아 둔 데이터 질문은 여기서 버려진다(서버는 출생 연도 질문만 싣는다)
  convo.clarify = data.clarify || null;
  convo.clarifyDeferred = Boolean(data.clarify_deferred);
  convo.explain = data.explain || null;
  explainOpen.clear(); // 새 결과다 — 앞 목록에서 펼쳐 둔 곡은 여기 없다
  // 출생 연도 질문은 결과와 함께 바로 띄운다(위 '예외'). 건너뛴 뒤에는 거절 버튼만 보인다
  panelMode = isBirthYearQuestion() && !convo.birthYearSkipped && convo.results.length > 0 ? "asking" : "idle";
  markHits(convo.results.map((r) => String(r.id)));
  renderPanel();
  resultList.scrollTop = 0; // 새 결과는 1위부터 보이게
}

function shownIds() {
  return convo ? convo.results.map((r) => String(r.id)) : [];
}

function isBirthYearQuestion() {
  return Boolean(convo && convo.clarify && convo.clarify.slot === "birth_year");
}

/* 출생 연도 질문을 이 검색에서는 더 묻지 않는다 — "잘 모르겠어요"·닫기·Esc.
   물어본 슬롯으로 적어 두면 다음 요청(질문 조회·거절)에 실려 가 서버도 다시 묻지 않는다 */
function dismissBirthYearQuestion() {
  convo.birthYearSkipped = true;
  if (!convo.askedSlots.includes("birth_year")) convo.askedSlots.push("birth_year");
  panelMode = "idle";
  renderPanel();
}

/*
 * 출생 연도 휠 — 슬롯처럼 세로로 돌려 한 해를 고른다. 스크롤 스냅으로 가운데 줄에 멈추고(마우스 휠·트랙패드·터치·
 * 방향키·클릭 모두 네이티브 스크롤), 가운데 줄의 연도가 값이다. 처음 열릴 때 위에서 기본 연도까지 굴러 내려온다.
 * 줄 높이는 style.css의 .wheel li와 같아야 한다.
 */
const WHEEL_ROW = 36;
const WHEEL_PAD = 2;                 // 가운데 줄이 첫·끝 연도도 가리킬 수 있게 위아래 빈 줄
const WHEEL_DEFAULT_YEAR = 2000;

function yearWheel(options) {
  const seen = (options || []).flatMap((o) => (String(o.value || "").match(/(?:19|20)\d{2}/g) || []).map(Number));
  const first = seen.length ? Math.min(...seen) : 1970;
  const last = seen.length ? Math.max(...seen) : 2015;
  const years = [];
  for (let y = first; y <= last; y += 1) years.push(y);
  const clamp = (y) => Math.min(last, Math.max(first, y));
  const start = clamp(convo.wheelYear || loadBirthYear() || WHEEL_DEFAULT_YEAR);

  const el = makeEl("div", "wheel");
  el.setAttribute("role", "group");
  el.setAttribute("aria-label", "출생 연도 고르기");
  const list = makeEl("ul", "wheellist");
  list.tabIndex = 0;
  list.setAttribute("aria-label", "연도를 돌려서 고르세요");
  for (let i = 0; i < WHEEL_PAD; i += 1) list.append(makeEl("li", "wheelpad"));
  for (const y of years) {
    const li = makeEl("li", "", String(y));
    li.dataset.year = String(y);
    li.addEventListener("click", () => list.scrollTo({ top: (y - first) * WHEEL_ROW, behavior: "smooth" }));
    list.append(li);
  }
  for (let i = 0; i < WHEEL_PAD; i += 1) list.append(makeEl("li", "wheelpad"));
  const label = makeEl("div", "wheelvalue");
  el.append(list, label);

  const index = () => Math.min(years.length - 1, Math.max(0, Math.round(list.scrollTop / WHEEL_ROW)));
  const value = () => years[index()];
  let raf = 0;
  const paint = () => {
    raf = 0;
    const y = value();
    convo.wheelYear = y; // 다시 그려져도(로딩 등) 돌려 둔 자리를 잃지 않게
    label.textContent = `${y}년생`;
    list.querySelectorAll("li[data-year]").forEach((li) => li.classList.toggle("active", Number(li.dataset.year) === y));
  };
  list.addEventListener("scroll", () => {
    if (!raf) raf = requestAnimationFrame(paint);
  });
  list.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    list.scrollTo({ top: (index() + (e.key === "ArrowDown" ? 1 : -1)) * WHEEL_ROW, behavior: "smooth" });
  });
  // 붙은 뒤에 위치를 잡는다 — 처음엔 맨 위에서 기본 연도까지 굴러 내려오고, 이미 돌려 둔 값이면 바로 그 자리
  requestAnimationFrame(() => {
    const top = (start - first) * WHEEL_ROW;
    if (convo.wheelYear) {
      list.scrollTop = top;
    } else {
      list.scrollTop = 0;
      requestAnimationFrame(() => list.scrollTo({ top, behavior: "smooth" }));
    }
    paint();
  });
  return { el, value };
}

/* 출생 연도 답 — 저장하고 같은 질의를 기존 분석 + birth_year로 다시 검색한다. 거절할 곡이 없으므로 턴을 쌓지 않고,
   실패하면 앞의 결과·대화를 그대로 둔다(답은 저장됐으므로 다음 검색부터 반영된다) */
async function answerBirthYear(value) {
  if (!convo || isBusy()) return;
  const year = parseBirthYear(value);
  if (!year) {
    dismissBirthYearQuestion();
    return;
  }
  saveBirthYear(year);
  const seq = searchSeq;
  const current = convo;
  current.birthYearSkipped = true; // 성공하든 실패하든 이 검색에서 다시 묻지 않는다
  panelMode = "loading";
  renderPanel();
  const body = { query: current.query, top_k: TOP_K, explain: true, birth_year: year, turn: current.turn, defer_clarify: true };
  if (current.analysis) body.prior_analysis = current.analysis; // 재분석(Gemini 호출) 없이 창만 넣어 다시 찾는다
  try {
    const data = await requestSearch(body);
    if (seq !== searchSeq || convo !== current) return; // 그 사이 새 검색을 했다
    showStatus("");
    applyResponse(data);
  } catch (err) {
    if (seq !== searchSeq || convo !== current) return;
    panelMode = "idle";
    renderPanel(); // 앞의 결과를 그대로 보여준다
    showStatus(`${searchFailureText(err)} 시기를 반영한 재검색은 실패해 앞의 결과를 그대로 둡니다.`, "warn");
  }
}

function canRejectMore() {
  if (!convo || convo.results.length === 0) return false;
  const next = new Set([...convo.rejectedIds, ...shownIds()]);
  return next.size <= MAX_REJECTED;
}

function makeEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(className, text, onClick) {
  const b = makeEl("button", className, text);
  b.type = "button";
  b.disabled = isBusy();
  b.addEventListener("click", (e) => {
    e.stopPropagation(); // document 핸들러가 곡 카드를 닫지 않게
    onClick();
  });
  return b;
}

function renderPanel() {
  updateLegendVisibility();
  if (!convo) {
    resultBlock.hidden = true;
    clarifyBox.hidden = true;
    runNote.hidden = true;
    syncRailMetrics();
    return;
  }
  resultBlock.hidden = false;
  clarifyBox.hidden = false;
  const sub = [];
  if (convo.results.length) sub.push(`${convo.results.length}곡`);
  if (convo.turn > 1) sub.push(`${convo.turn - 1}번째 다시 찾음`);
  resultSub.textContent = sub.join(" · ");
  renderRunNote();

  const frag = document.createDocumentFragment();
  convo.results.forEach((r, i) => {
    const id = String(r.id);
    const song = songsById.get(id);
    const li = makeEl("li", "resultitem");
    li.dataset.id = id;
    if (!song) li.classList.add("offmap");
    li.append(makeEl("span", "resultrank", String(i + 1)));

    const cover = makeEl("img", "resultcover");
    cover.alt = "";
    cover.loading = "lazy";
    cover.decoding = "async";
    /*
     * 검색 응답의 cover_url을 쓰고, 없으면 맵 데이터에서 찾는다.
     * src를 빈 문자열로 두면 브라우저가 페이지 자신을 다시 내려받으므로
     * 주소가 없을 때는 아예 넣지 않는다 (CSS의 회색 사각형만 남는다).
     */
    const url = r.cover_url || (song && song.c) || "";
    if (url) cover.src = url;
    li.append(cover);

    const text = makeEl("span", "resulttext");
    text.append(makeEl("span", "resultsong", r.title));
    const meta = makeEl("span", "resultmeta");
    meta.append(makeEl("span", "resultartist", r.artist || ""));
    // 장르는 지도와 같은 색으로 칠해 두 화면이 같은 기준임을 보인다
    const genre = r.genre || (song && song.g) || "";
    if (genre) {
      const tag = makeEl("span", "resultgenre", genre);
      tag.style.color = GENRE_COLORS[genre] || FALLBACK_COLOR;
      meta.append(tag);
    }
    text.append(meta);
    li.append(text);

    /*
     * 근거는 서버가 기록을 실어 보낸 곡에만 붙는다. 의미 검색 서버에 닿지
     * 못해 문자열 검색으로 대체했을 때는 설명할 기록 자체가 없다.
     */
    const x = r.explain;
    if (x) li.append(explainToggle(id, x));
    frag.appendChild(li);
    // 다시 그리기 전에 펼쳐 둔 곡은 펼친 채로 되돌린다
    if (x && explainOpen.has(id)) frag.appendChild(explainPanel(x));
  });
  resultList.replaceChildren(frag);

  clarifyBox.replaceChildren(...clarifyContent());
  // 좁은 화면에서는 결과가 늘면 레일 높이가 바뀐다 — 지도가 그만큼 내려가야 한다
  syncRailMetrics();
}

function clarifyContent() {
  if (panelMode === "question_loading") {
    return [makeEl("p", "clarifynote", "추가 질문을 준비하는 중…")];
  }
  if (panelMode === "loading") {
    return [makeEl("p", "clarifynote", "다시 찾는 중…")];
  }
  if (convo.results.length === 0) {
    return [makeEl("p", "clarifynote", "더 보여드릴 후보가 없어요. 기억나는 다른 단서로 다시 검색해 보세요.")];
  }
  if (panelMode === "asking" && convo.clarify) {
    const q = convo.clarify;
    const bubble = makeEl("div", "bubble");
    bubble.append(makeEl("p", "bubbleq", q.question));
    const options = makeEl("div", "options");
    const aboutUser = q.slot === "birth_year"; // 곡이 아니라 사용자에 대한 질문 — 후보 수가 없고, 거절 없이 바로 다시 검색한다
    if (aboutUser) {
      // 밴드 버튼 대신 돌려서 고르는 연도 휠. 서버의 밴드 목록은 연도 범위로만 쓴다(답은 '1998' 한 해로 보낸다)
      const wheel = yearWheel(q.options);
      bubble.append(wheel.el);
      bubble.append(makeEl("p", "clarifynote", "답하면 그 시기의 곡을 먼저 보여드려요. 이 브라우저에만 저장돼요. 다음부터는 묻지 않아요."));
      options.append(button("option primary", "이 연도로 찾기", () => answerBirthYear(String(wheel.value()))));
    } else {
      for (const opt of q.options) {
        const b = button("option", "", () => submitTurn({ slot: q.slot, value: opt.value, skipped: false }));
        b.append(makeEl("span", "", opt.value));
        b.append(makeEl("span", "optioncount", `${opt.count}곡`));
        options.append(b);
      }
    }
    options.append(
      button("option skip", "잘 모르겠어요", () =>
        aboutUser ? dismissBirthYearQuestion() : submitTurn({ slot: q.slot, value: "", skipped: true })
      )
    );
    bubble.append(options);
    const cancel = button("linkbtn", aboutUser ? "그냥 결과 볼게요" : "결과로 돌아가기", () => {
      if (aboutUser) {
        dismissBirthYearQuestion();
        return;
      }
      panelMode = "idle";
      renderPanel();
    });
    return [bubble, cancel];
  }
  if (canRejectMore()) {
    return [button("rejectbtn", "이 중에는 없어요", onReject)];
  }
  return [makeEl("p", "clarifynote", "여기까지 찾아봤어요. 기억나는 다른 단서로 다시 검색해 보세요.")];
}

/* ---------- 선정 근거 ----------
 *
 * 서버가 explain=true 응답에 함께 실어 보낸 기록만 그린다. 여기서 점수를 다시
 * 계산하거나 문장을 짓지 않는다 — 경로·규칙 이름, 단계 이름, 요약문, 가사 일치
 * 유형까지 전부 응답에 들어 있는 것을 그대로 쓴다. 화면이 제 나름대로 말을
 * 만들기 시작하면 "가사 원문에 그대로 있다"처럼 기록보다 강한 주장이 섞인다.
 * 실제로 그 문장은 표기 정규화 후의 일치를 원문 일치로 잘못 말한 것이었다.
 *
 * 누를 때 다시 검색하지 않는 이유: 질의 분석이 같은 입력에도 흔들려서, 다시
 * 물으면 지금 화면에 뜬 목록이 아니라 다른 실행을 설명하게 된다.
 */

// 펼쳐 둔 곡. 목록을 다시 그려도 펼친 상태가 남게 화면 밖에 둔다
const explainOpen = new Set();

// 서버의 _fmt와 같은 표기여야 요약문과 표의 숫자가 어긋나 보이지 않는다
function exDelta(v) {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
}

function exGroup(title, note) {
  const g = makeEl("div", "exgroup");
  const head = makeEl("h4", "exhead", title);
  // 제목 옆 단서 — "계측용", "최종 점수에 없음"처럼 표를 잘못 읽지 않게 하는 말
  if (note) head.append(makeEl("span", "exnote", note));
  g.append(head);
  return g;
}

/* 이름(+부연)과 증감 한 줄. 감점은 색으로도 구분한다 */
function exItem(label, detail, delta) {
  const row = makeEl("div", "exrow");
  const left = makeEl("span", "exlabel", label);
  if (detail) left.append(makeEl("span", "exdetail", detail));
  const num = makeEl("span", "exnum", exDelta(delta));
  if (delta < 0) num.classList.add("minus");
  row.append(left, num);
  return row;
}

/* 증감이 아니라 사실 하나를 적는 줄 (순위·전략·가중치 같은 것) */
function exFact(label, value, detail) {
  const row = makeEl("div", "exrow");
  const left = makeEl("span", "exlabel", label);
  if (detail) left.append(makeEl("span", "exdetail", detail));
  row.append(left, makeEl("span", "exfact", value));
  return row;
}

function exSum(value) {
  const row = makeEl("div", "exrow exsum");
  row.append(makeEl("span", "exlabel", "합"), makeEl("span", "exnum", value));
  return row;
}

function explainToggle(id, x) {
  const b = makeEl("button", "exbtn", "근거");
  b.type = "button";
  b.title = "이 곡이 이 자리에 온 근거";
  b.setAttribute("aria-expanded", String(explainOpen.has(id)));
  b.addEventListener("click", (e) => {
    e.stopPropagation(); // 결과 항목의 클릭(지도 카드 열기)까지 가지 않게
    toggleExplain(id, b, x);
  });
  return b;
}

function toggleExplain(id, btn, x) {
  const li = btn.closest(".resultitem");
  const open = explainOpen.has(id);
  if (open) {
    explainOpen.delete(id);
    const panel = li.nextElementSibling;
    if (panel && panel.classList.contains("explainrow")) panel.remove();
  } else {
    explainOpen.add(id);
    li.after(explainPanel(x));
  }
  btn.setAttribute("aria-expanded", String(!open));
  syncRailMetrics(); // 좁은 화면에서는 레일이 높아진다 — 지도가 그만큼 내려간다
}

function explainPanel(x) {
  const li = makeEl("li", "explainrow");
  const box = makeEl("div", "explain");

  // 서버가 기록만으로 만든 문장. 화면이 고쳐 쓰지 않는다
  box.append(makeEl("p", "exsummary", x.summary));

  // 이 곡이 어디서 어디로 갔는가. 아래는 전부 그 이유다
  if (x.rank_before_rerank != null && x.rank_after_rerank != null) {
    box.append(
      exFact("순위", `${x.rank_before_rerank}위 → ${x.rank_after_rerank}위`, "리랭킹 전 → 최종")
    );
  }

  /*
   * 순서 규칙이 점수표보다 먼저 온다. 점수와 무관하게 자리를 정하는 규칙이라,
   * 점수를 먼저 보여주면 그 점수가 순서를 정했다고 읽힌다.
   */
  if (x.order_rules.length) {
    const g = exGroup("순서 규칙", "점수와 무관하게 순서만 바꿈");
    for (const rule of x.order_rules) {
      const row = makeEl("div", "exrow");
      const left = makeEl("span", "exlabel", rule.label);
      if (rule.detail) left.append(makeEl("span", "exdetail", rule.detail));
      row.append(left);
      g.append(row);
    }
    box.append(g);
  }

  /*
   * 경로 기여 + 가감점 = 검색 점수. 합이 실제로 맞는 장부다 — 그래야 보는
   * 사람이 "이 줄들이 정말 저 점수를 만들었나"를 직접 확인할 수 있다.
   */
  if (x.paths.length) {
    const g = exGroup("경로 기여");
    for (const path of x.paths) g.append(exItem(path.label, exPathDetail(path), path.delta));
    g.append(exSum(exDelta(x.path_total)));
    box.append(g);
  }
  if (x.adjustments.length) {
    const g = exGroup("가감점");
    for (const adj of x.adjustments) g.append(exItem(adj.label, adj.detail, adj.delta));
    g.append(exSum(exDelta(x.adjustment_total)));
    box.append(g);
  }
  if (x.retrieval_score != null) {
    /*
     * 위 두 합이 이 값을 만든다. 다만 각 줄은 넷째 자리에서 반올림해 적으므로
     * 눈으로 더한 값이 마지막 자리에서 1 어긋날 수 있다. 검산하는 사람이
     * 계산이 틀렸다고 읽지 않게 밝혀 둔다.
     */
    const total = exFact(
      "검색 점수", x.retrieval_score.toFixed(4), "경로 기여 + 가감점 · 표시값은 반올림"
    );
    total.classList.add("extotal");
    box.append(total);
  }

  // 1차 절단에서 탈락한 기여. 위 합계에 넣으면 설명이 실제 점수보다 커진다
  if (x.dropped_paths.length) {
    const g = exGroup("반영 안 된 기여", "1차 절단에서 탈락 · 최종 점수에 없음");
    for (const path of x.dropped_paths) {
      const row = exItem(path.label, exPathDetail(path), path.delta);
      row.classList.add("dim");
      g.append(row);
    }
    box.append(g);
  }

  // 순위가 그대로면 적을 것이 없다 — 가사 경로가 돌았다는 사실만으로는 근거가 아니다
  if (
    x.lyric_boost_rank_before != null && x.lyric_boost_rank_after != null &&
    x.lyric_boost_rank_before !== x.lyric_boost_rank_after
  ) {
    const g = exGroup("가사 부스트");
    g.append(
      exFact(
        "융합 순위",
        `${x.lyric_boost_rank_before}위 → ${x.lyric_boost_rank_after}위`,
        "가산점 직전 → 직후"
      )
    );
    box.append(g);
  }

  if (x.score_mix) box.append(explainMix(x));
  if (x.lyric_match) box.append(explainLyric(x));
  for (const item of x.evidence || []) box.append(explainEvidence(item));
  if (x.model_notes.length) box.append(explainModelNotes(x));

  li.append(box);
  return li;
}

function exPathDetail(path) {
  return [`${path.rank}위`, path.detail].filter(Boolean).join(" · ");
}

function explainMix(x) {
  const mix = x.score_mix;
  const g = exGroup("리랭킹", x.reorder_stage_label);

  /*
   * 묶음 안 순위만 모델에 귀속할 수 있다. 전체 순위 변화에는 순서 규칙이 섞여
   * 있어서, 그것을 모델이 한 일로 적으면 하지 않은 일을 말하게 된다.
   * 순서를 합성 점수가 정하지 않은 곡(reordered=false)에는 아예 쓰지 않는다.
   */
  if (
    x.reorder_applied && mix.reordered &&
    x.group_rank_before != null && x.group_rank_after != null
  ) {
    g.append(
      exFact(
        "묶음 안 순위",
        `${x.group_rank_before}위 → ${x.group_rank_after}위`,
        "모델에 귀속되는 유일한 변화"
      )
    );
  }
  if (mix.strategy) g.append(exFact("전략", mix.strategy));
  g.append(
    exFact(
      "리랭커 가중치",
      mix.weight.toFixed(3),
      mix.confidence < 1
        ? `설정 ${mix.configured_weight.toFixed(2)} · 신뢰도 ${mix.confidence.toFixed(3)}`
        : "설정값 그대로"
    )
  );
  g.append(
    exFact(
      "합성",
      mix.final.toFixed(3),
      `리랭커 ${mix.rerank_component.toFixed(3)} · 검색 ${mix.retrieval_component.toFixed(3)}`
    )
  );

  // 점수는 합성해도 순서는 검색 순서를 그대로 둔 곡이 절반 가까이 된다
  if (!mix.reordered) {
    g.append(makeEl("p", "exwarn", "점수는 합성했지만 순서는 검색 순서를 그대로 두었습니다."));
  }
  if (!mix.rerank_normalized) {
    g.append(makeEl("p", "exwarn", "리랭커 원점수를 정규화 없이 섞었습니다."));
  }
  // 하락 폭이 아니다 — 보호 곡이 원래 앞에 있었으면 순위는 그대로다
  if (x.behind_protected > 0) {
    g.append(
      makeEl(
        "p",
        "exwarn",
        `앞에 가사 보호 곡 ${x.behind_protected}곡이 놓였습니다. 이 곡이 그만큼 밀렸다는 뜻은 아닙니다.`
      )
    );
  }
  return g;
}

function explainLyric(x) {
  const m = x.lyric_match;
  const g = exGroup("가사 일치", "계측용 · 순위 계산에 쓰이지 않음");
  // 일치 유형을 뭐라 부를지는 서버가 정한다 (표기 정규화 후의 일치다)
  if (x.lyric_match_label) g.append(makeEl("p", "exlyricnote", x.lyric_match_label));
  g.append(exFact("기억한 구절", m.clue_text, `${m.clue_kind} 단서`));
  g.append(
    exFact("일치한 표기", m.matched_phrase, m.is_variant ? "원문이 아니라 음차 변형 후보" : "")
  );
  g.append(exFact("단서 확신도", m.confidence.toFixed(2)));
  // 짧을수록 우연히 겹칠 확률이 높다. 겹친 곡 수와 함께 봐야 뜻이 산다
  g.append(exFact("정규화 길이", `${m.normalized_length}자`));
  g.append(
    exFact("이 구절을 가진 곡", `${m.corpus_match_count}곡`, "후보 절단 전 전체 가사에서")
  );
  return g;
}

/*
 * 원문 인용. 여기서 문장을 만들지 않는다 — 서버가 원문에서 글자 단위로 떠 온
 * 구간과, 그것이 어디서 왔는지(출처·위치·판)를 그대로 옮긴다.
 *
 * 인용이 없으면 이 함수는 불리지 않는다. 되짚지 못한 자리를 그럴듯한 문장으로
 * 채우면 "가사에 이렇게 있다"가 근거 없는 주장이 된다.
 */
function explainEvidence(item) {
  // 순위를 만든 근거인지, 뒤에 덧붙인 설명인지. 이름은 서버가 정한다
  const g = exGroup("원문 근거", item.kind_label);

  // 구절만 떼면 앞뒤가 잘린다. 줄 전체를 보이고 일치한 자리만 표시한다
  const quote = makeEl("p", "exquote");
  const at = item.line.indexOf(item.quote);
  if (at < 0) {
    quote.textContent = item.line;
  } else {
    quote.append(
      document.createTextNode(item.line.slice(0, at)),
      makeEl("mark", "exquotehit", item.quote),
      document.createTextNode(item.line.slice(at + item.quote.length))
    );
  }
  g.append(quote);

  // 되짚을 수 있어야 근거다 — 어느 필드의 몇 번째 글자이고, 어느 판인가
  g.append(exFact("출처", item.source, `${item.char_start}–${item.char_end}자`));
  g.append(exFact("가사 판", item.data_version));
  return g;
}

function explainModelNotes(x) {
  const g = exGroup("모델이 쓴 이유");
  // 검증된 기록과 한 덩어리로 읽히지 않게, 서버가 함께 보낸 경고를 그대로 붙인다
  g.append(makeEl("p", "exwarn", x.model_notes_caption));
  for (const text of x.model_notes) g.append(makeEl("p", "exmodelnote", text));
  return g;
}

/*
 * 요청 한 번의 실행 기록. 곡별 근거와 달리 질의 전체에 해당한다 — 리랭커가
 * 예외로 멈췄으면 열 곡의 이야기가 통째로 달라지므로, 곡을 펼쳐야 보이는
 * 자리에 두지 않는다.
 */
function renderRunNote() {
  const x = convo && convo.explain;
  if (!x) {
    runNote.hidden = true;
    return;
  }
  const parts = [x.reorder_stage_label || x.reorder_stage];
  if (x.reranker && x.reranker !== "none") parts.push(x.reranker);
  // 호출 수 ≠ 성공 수다. 같을 때는 말할 것이 없다
  if (x.rerank_calls !== x.rerank_calls_applied) {
    parts.push(`${x.rerank_calls}번 불러 ${x.rerank_calls_applied}번 적용`);
  }
  if (x.rerank_error) parts.push(x.rerank_error);
  /*
   * 질의 분석이 규칙 폴백이면 결과의 성격이 통째로 달라진다(쿼터 초과·시간 초과).
   * 검색은 그대로 돌아서 결과만 봐서는 구분되지 않으므로 여기서 말한다.
   * 문구는 서버가 보낸 것을 그대로 쓴다 — 화면이 짓지 않는다.
   */
  if (x.analysis_fallback) parts.push(x.analysis_fallback_label);
  /*
   * 예외로 죽은 경로. 가중치는 남아 있으므로 이것이 없으면 "가중치 0.20인데
   * 기여가 한 줄도 없다"가 되어, 경로가 돌았지만 이 곡을 못 올린 것으로 읽힌다.
   */
  for (const path of x.failed_paths || []) parts.push(path.label);
  runNote.textContent = parts.filter(Boolean).join(" · ");
  runNote.classList.toggle(
    "warn",
    !!(x.rerank_error || x.analysis_fallback || (x.failed_paths || []).length)
  );
  runNote.hidden = false;
}

/* ---------- 레일 접기 ----------
 * 지도를 넓게 보고 싶을 때 레일을 화면 밖으로 밀어낸다. 새로고침해도 유지한다.
 * 넓은 화면에서는 레일 전체가, 좁은 화면(위로 누운 상태)에서는 결과·재질문만
 * 접히고 검색줄은 남는다 — 어느 쪽인지는 CSS가 나눈다.
 *
 * 예전에는 결과 패널을 지도 위에 띄우고 끌어 옮길 수 있게 했다. 패널이 곡을
 * 가려서였는데, 레일은 지도 영역 밖에 앉으므로 그 기능이 필요 없어졌다.
 */
const RAIL_CLOSED_KEY = "vf.railClosed";
/*
 * 접기 단축키는 Cmd/Ctrl + \ 다. 고른 이유:
 *  - 조합키라 검색창에 커서가 있어도 동작한다. 맨 글자 단축키였다면
 *    검색어를 치는 동안 레일이 접혔다 펴진다.
 *  - Cmd/Ctrl+B는 파이어폭스에서 북마크 사이드바를 연다.
 *  - e.code로 보므로 한글 입력 상태(같은 자리가 ₩)에서도 같은 키로 잡힌다.
 */
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.userAgent);
const RAIL_KEY_HINT = IS_MAC ? "\u2318\\" : "Ctrl+\\";

function updateRailToggleLabel() {
  const closed = document.body.classList.contains("railclosed");
  railToggle.title = `${closed ? "펼치기" : "접기"} (${RAIL_KEY_HINT})`;
  railToggle.setAttribute("aria-expanded", String(!closed));
}
railToggle.setAttribute("aria-keyshortcuts", IS_MAC ? "Meta+\\" : "Control+\\");
updateRailToggleLabel();

function setRailClosed(closed) {
  // 새 검색마다 불린다. 상태가 그대로면 곡 전체를 다시 배치할 이유가 없다
  if (document.body.classList.contains("railclosed") === closed) return;
  document.body.classList.toggle("railclosed", closed);
  updateRailToggleLabel();
  // 지도가 쓸 수 있는 폭이 달라졌다 — 배율과 곡 위치를 다시 잡는다
  placeSongs();
  try {
    localStorage.setItem(RAIL_CLOSED_KEY, closed ? "1" : "");
  } catch {
    /* 저장이 막혀도 이번 세션 상태는 유지된다 */
  }
}

railToggle.addEventListener("click", (e) => {
  e.stopPropagation();
  setRailClosed(!document.body.classList.contains("railclosed"));
});

stackedRail.addEventListener("change", () => {
  // 세로 레일 ↔ 가로 레일로 바뀌면 지도가 비켜 앉을 방향이 달라진다
  placeSongs();
});

try {
  if (localStorage.getItem(RAIL_CLOSED_KEY)) setRailClosed(true);
} catch {
  /* 읽을 수 없으면 펼친 상태로 시작한다 */
}

function onReject() {
  if (!canRejectMore() || isBusy()) return;
  if (convo.clarify && !isBirthYearQuestion()) {
    // 이 결과로 이미 받아 둔 질문이다(닫았다가 다시 눌렀다) — 다시 조회하지 않는다
    panelMode = "asking";
    renderPanel();
    return;
  }
  if (convo.clarifyDeferred) return requestClarification();
  return submitTurn(null); // 물을 게 없으면 거절만 한다
}

const CLARIFY_FAILED_NOTE = "추가 질문을 준비하지 못해 질문 없이 다음 곡을 찾았습니다.";

/* 지금 후보로 물을 것이 있는지 서버에 묻는다. 질문만 가져온다 — 턴·거절·결과는 그대로다.
   질문이 없거나 조회가 실패(오류·제한 시간)하면 거절만 해서 다음 결과로 간다. 실패한 조회는 다시 시도하지
   않는다(클릭 한 번에 조회 한 번·거절 검색 한 번). 그 거절 검색까지 실패하면 submitTurn이 앞의 결과를 그대로 둔다 */
async function requestClarification() {
  const current = convo;
  const seq = searchSeq;
  // 답을 실어 보낼 분석이 없으면 질문을 받아도 반영하지 못한다(서버는 prior_analysis 없는 answers를 거부한다)
  if (!current.analysis) return submitTurn(null);
  panelMode = "question_loading";
  renderPanel();
  let question = null;
  let failed = false;
  try {
    const data = await requestSearch({
      asked_slots: current.askedSlots,
      previous_candidate_ids: current.candidateIds.slice(0, MAX_PREVIOUS_CANDIDATES),
      shown_ids: shownIds(),
      rejected_ids: current.rejectedIds,
      turn: current.turn,
    }, CLARIFY_PATH, CLARIFY_TIMEOUT_MS);
    question = data.clarify || null;
  } catch (err) {
    failed = true;
  }
  if (seq !== searchSeq || convo !== current) return; // 그 사이 새 검색을 했다 — 늦은 응답은 버린다
  panelMode = "idle";
  if (question) {
    current.clarify = question;
    showStatus("");
    panelMode = "asking";
    renderPanel();
    return;
  }
  return submitTurn(null, failed ? CLARIFY_FAILED_NOTE : "");
}

/* 보여준 곡을 거절하고(답이 있으면 함께 실어) 다음 결과를 찾는다. note는 성공했을 때 상태줄에 남길 안내 */
async function submitTurn(answer, note = "") {
  if (!convo || isBusy()) return;
  const current = convo;
  const rejected = [...new Set([...convo.rejectedIds, ...shownIds()])];
  let answers = answer ? [...convo.answers, answer] : convo.answers;
  if (!convo.analysis) answers = []; // 서버는 prior_analysis 없는 answers를 거부한다

  if (answer && answer.slot === "birth_year" && !answer.skipped) {
    const year = parseBirthYear(answer.value);
    if (year) saveBirthYear(year); // 다음 검색부터는 묻지 않고 요청에 실어 보낸다
  }

  const body = {
    query: convo.query,
    top_k: TOP_K,
    prior_analysis: convo.analysis,
    answers,
    asked_slots: convo.askedSlots,
    previous_candidate_ids: convo.candidateIds.slice(0, MAX_PREVIOUS_CANDIDATES),
    rejected_ids: rejected,
    turn: convo.turn,
    explain: true,
    defer_clarify: true,
    birth_year: loadBirthYear(),
  };

  const seq = searchSeq;
  const previousMode = panelMode;
  panelMode = "loading";
  renderPanel();
  try {
    const data = await requestSearch(body);
    if (seq !== searchSeq || convo !== current) return; // 그 사이 새 검색을 했다
    convo.answers = answers;
    showStatus(note, note ? "warn" : undefined);
    applyResponse(data);
  } catch (err) {
    if (seq !== searchSeq || convo !== current) return;
    // 실패하면 상태를 건드리지 않고 직전 화면으로 되돌린다 — 다시 누르면 재시도된다
    panelMode = previousMode;
    renderPanel();
    showStatus(`${searchFailureText(err)} 잠시 후 다시 시도해 주세요.`, "warn");
  }
}

resultList.addEventListener("click", (e) => {
  const li = e.target.closest(".resultitem");
  if (!li) return;
  e.stopPropagation(); // document 핸들러가 방금 연 카드를 닫지 않게
  const song = songsById.get(li.dataset.id);
  if (!song) return; // 지도에 없는 곡
  showMapView();
  openCard(li.dataset.id);
  song.el.scrollIntoView({ block: "center", inline: "center" });
});
rail.addEventListener("click", (e) => e.stopPropagation());

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = searchInput.value.trim();
  if (!q) {
    clearSearch();
    return;
  }
  runSearch(q);
});

/* 처음부터 검색한다(검색줄 제출). 출생 연도 답 뒤의 재검색은 answerBirthYear가 기존 분석으로 따로 한다 */
async function runSearch(q) {
  const seq = ++searchSeq;
  convo = null;
  panelMode = "idle"; // 앞 대화의 요청이 나가 있었더라도 그 응답은 버려진다 — 로딩 표시를 남기지 않는다
  renderPanel();
  clearBtn.hidden = false;
  showStatus("검색 중…");
  try {
    const data = await requestSearch({ query: q, top_k: TOP_K, explain: true, birth_year: loadBirthYear(), defer_clarify: true });
    if (seq !== searchSeq) return;
    showStatus("");
    startConvo(q, data);
    if (convo.results.length === 0) showStatus("일치하는 곡이 없습니다.");
  } catch (err) {
    if (seq !== searchSeq) return;
    // 문자열 검색에는 거절·재질문을 붙이지 않는다 — 서버 상태가 없다
    const n = markHits(localSearch(q));
    const why = searchFailureText(err);
    showStatus(
      n === 0
        ? `${why} 제목·가수 문자열 검색으로도 찾지 못했습니다.`
        : `${why} 제목·가수 문자열 검색 결과를 표시합니다.`,
      "warn"
    );
  }
}

clearBtn.addEventListener("click", clearSearch);
document.addEventListener("keydown", (e) => {
  // 누르고 있으면 곡 전체를 반복해서 다시 배치하게 되므로 자동 반복은 무시한다
  if (
    (e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey &&
    e.code === "Backslash" && !e.repeat
  ) {
    e.preventDefault();
    setRailClosed(!document.body.classList.contains("railclosed"));
    return;
  }
  if (e.key !== "Escape") return;
  if (panelMode === "asking") {
    // 질문 중 Esc는 검색 전체가 아니라 질문만 닫는다
    if (isBirthYearQuestion()) {
      dismissBirthYearQuestion();
      return;
    }
    panelMode = "idle";
    renderPanel();
    return;
  }
  clearSearch();
});
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    searchForm.requestSubmit();
  }
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    placeSongs();
    if (cardId && !card.hidden) openCard(cardId); // 카드 위치도 다시 잡는다
  }, 150);
});

init();
