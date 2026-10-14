"""리랭킹을 **일부러 부르지 않는** 정책. 서비스(`search_router`)와 평가(`evaluate_search_accuracy`)가 함께 쓴다.

왜 한 곳에 두는가: 보호 규칙처럼 두 곳에 사본으로 두면 한쪽만 고쳐져 서비스와 측정이
다른 규칙으로 돈다(`select_protected_lyric_ids`가 같은 이유로 한 곳에 있다).

두 가지 사유가 있다.

1. 표지 지배 질의(`RERANK_SKIP_IMAGE_DOMINANT`) — SigLIP2가 실제 표지를 보고 만든 순서를,
   표지를 보지 않는 텍스트 리랭커가 다시 섞지 않게 한다. 기존 정책 그대로다.
2. 구체적인 뮤직비디오 장면 질의인데 그 장면을 판단할 출처 자료가 없는 경우
   (`RERANK_SKIP_MV_SCENE_UNVERIFIABLE`). 2026-10-11 사례: "뮤비에 의자가 하늘에 떠있는
   장면"에 리랭커가 "옥상에서 의자를 던지는 장면"이라는, 어떤 입력에도 없는 장면을 이유로
   썼다. 가사 요약의 옥상과 질의의 의자를 결합한 것으로 보인다. 모델에게 "지어내지 마"라고
   적는 대신, **판단할 자료가 없는 질의에서는 모델의 추측으로 순서를 더 바꾸지 않는다.**

정책이 **하지 않는 것**:
- 특정 곡을 막거나 내리지 않는다. 자료 없음은 장면이 없다는 증거가 아니다.
- 후보 검색·답변 보정·거절 제외는 그대로다. 생략한 요청의 최종 순서는 그 후보 순서다
  (MV 생략은 가사 exact 보호 배치만 모델 없이 적용한다 — `search_router.place_protected_lyrics_first`).
- 리랭커 프롬프트·질의 분석 프롬프트를 바꾸지 않는다(바꾸면 모든 질의의 기준선이 바뀐다).
- Cross-Encoder 등 다른 백엔드는 건드리지 않는다. 리랭커 클래스가 `skips_*` 속성으로
  동의한 경우에만 적용한다.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

from src.backend.schemas.query import QueryAnalysis
from src.retrieval.explain import (
    RERANK_SKIP_IMAGE_DOMINANT,
    RERANK_SKIP_MV_SCENE_UNVERIFIABLE,
)

# ---------------------------------------------------------------------------
# 표지 지배 질의 — 기존 정책 (search_router에서 옮겨 왔다. 동작은 같다)
# ---------------------------------------------------------------------------


def should_skip_rerank_for_image(reranker: Any, analysis: QueryAnalysis) -> bool:
    """이미지 지배 질의는 텍스트 LLM 리랭킹을 건너뛴다.

    SigLIP2가 실제 앨범 커버를 보고 만든 순위를, 이미지를 직접 보지 않는
    listwise reranker가 다시 섞어 회귀시키는 것을 막는다.
    mixed 질의(image < 0.70)는 기존 리랭킹을 그대로 사용한다.

    리랭커가 `skips_image_dominant`를 켠 경우에만 적용한다. 기존 Cross-Encoder는
    v0.5 기준선부터 이미지 질의도 리랭킹해 왔으므로, 여기서 함께 끄면 백엔드를
    바꾸지 않은 사람의 검색 결과까지 달라진다.
    """
    if not getattr(reranker, "skips_image_dominant", False):
        return False
    return bool(
        analysis.has_visual_clue
        and float(analysis.modality_weights.image or 0.0) >= 0.70
    )


# ---------------------------------------------------------------------------
# 구체적인 MV 장면 질의 판정 — 원문만 본다(분석 프롬프트와 무관, 폴백·prior_analysis에서도 같다)
#
# 문형 기준이다. 뮤비 표지(뮤비·뮤직비디오·MV·M/V·엠브이)마다 **장면 구간**을 정하고 그 안만 본다.
#
#   표지 앞  : 표지를 직접 꾸미는 관형절일 때만("…떠 있는 뮤비", "…던지는 뮤직비디오"). 연결어미(인데/고/지만)나
#              쉼표로 끊긴 앞 절은 다른 화제다 — "남자 그룹이고 뮤비가 유튜브에 있어"의 남자는 장면이 아니다.
#   표지 뒤  : 표지에 붙은 말로 가른다.
#              · 내용을 여는 말(에서/에/속/마지막에/내내/배경/내용…) → 뒤 전부가 장면 구간
#              · 조사(가/는/도/를) + 첫 절이 장면을 담거나 "봤는데/기억나는데/인상적이었는데"처럼 이어지는 말 → 뒤 전부
#              · 그 외("뮤비가 유명해", "뮤비도 화제였고", "뮤비 찍다가") → 첫 절까지만. 뒤 절은 뮤비 내용이 아니다
#              가사·제목·멜로디 같은 다른 화제가 나오면 거기서 끊는다.
#   장면     : '장면/씬' | 화면 속성(흑백·세피아…) | 구체 명사(사물·장소·자연·인물·동물) + 시각 서술어/장면 구조어.
#              여는 말 뒤에서는 시각 서술어만으로도 장면이다("뮤비에서 편의점 앞에 앉아서 컵라면 먹어").
#   부정     : **표지에 바로 붙은** 기억 부정만 뮤비 부정이다("뮤비는 기억 안 나고", "뮤비 못 봤는데"). 다른 대상에 대한
#              부정("제목은 기억 안 나는데", "누가 불렀는지 모르겠는데")은 판정에 쓰지 않는다.
#              불확실은 장면 서술어에 붙었을 때만("떠 있었는지는 기억이 안 나"). 장면 자체의 부정("장면은 못 봤어")도 같다.
#
# 구체 명사를 요구하는 이유: 서술어 어간만으로는 "얼마·얼굴(얼)", "녹화(녹)", "타고난(타고)", "남달리(달리)",
# "조회수가 올라가(올라가)", "사랑에 빠진(빠지)"이 전부 장면이 된다.
# ---------------------------------------------------------------------------

# 뮤직비디오를 가리키는 말. "영상"·"무대"는 넣지 않는다 — 챌린지 영상·공연 장면은 MV 장면이 아니다.
_MV_MARKER_RE = re.compile(
    r"뮤직\s*비디오|뮤비|엠브이|(?<![A-Za-z])M\s*/?\s*V(?![A-Za-z])", re.IGNORECASE
)

# 따옴표 안은 가사·제목 인용이다. "가사에 '뮤비에서 의자를 던져'라는 말이 있어"의 뮤비는 장면이 아니다.
_QUOTED_RE = re.compile(r"[\"'“”‘’「」『』][^\"'“”‘’「」『』]{1,80}[\"'“”‘’「」『』]")

# 인용 부호 없이 낱말로 언급한 것 — "뮤비라는 말이 가사에 나와".
_MENTIONED_AS_WORD_RE = re.compile(
    r"(?:뮤직\s*비디오|뮤비|엠브이|(?<![A-Za-z])MV(?![A-Za-z]))\s*(?:라는|이라는|란|이란|라고|이라고)\s*(?:말|단어|가사|제목|부분|구절)",
    re.IGNORECASE,
)

# 문장 경계.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")
# 절 경계 — 쉼표, 연결어미, 그리고 **구두점 없이 끝난 종결어미** 뒤의 띄어쓰기(채팅식 "뮤비도 유명했어 남자 발라드야").
# "에서 "는 절 경계가 아니다("옥상에서 뛰어내리는").
_CLAUSE_BOUNDARY_RE = re.compile(
    r",|(?:인데|는데|은데|지만|고|며|면|했어|였어|었어|았어|있어|없어|이야|거야|더라|거든|같아|어요|네요|습니다)\s"
)

# 다른 화제로 넘어가는 말 — 여기서부터는 뮤비 장면이 아니다("뮤비가 유명한 노래인데 가사에 바다가 나와").
# 목소리·음색·보컬 **바로 앞의 남자/여자(들)(의)는 그 화제의 일부**다("춤추는 여자 목소리 노래" · "여자의 목소리" · "여자들 목소리"의 여자는
# 가수) — 구간을 그 앞에서 끊는다. 목소리 앞에서만 끊으면 "뮤비에서 춤추는 여자 |"의 여자가 화면 속 사람으로 남는다(2026-10-12 발견,
# §2-29 ②). 쉼표·다른 어절이 사이에 있으면 앞의 여자는 그대로 장면 속 인물이다("의자를 던져, 목소리는 여자였어").
_TOPIC_SWITCH_RE = re.compile(
    r"가사|제목|후렴|도입부|멜로디|(?:(?:남자|여자)들?의?\s*)?(?:목소리|음색|보컬)|노래\s*(?:는|가|인데|였|이었)|곡\s*(?:은|이|인데)"
)

# 표지 뒤에서 **내용을 여는 말**. 이 뒤는 뮤비 안의 이야기다. "에서부터"는 여는 말이 아니다(화제의 출발점).
_OPENER_RE = re.compile(
    r"^\s*(?:에서(?!부터)|에(?!게|\s*대해)|엔|속|안(?:에|에서)?|마지막|처음|첫|끝|초반|중반|후반|시작|내내|중간|도중|"
    r"배경|내용|장면|씬|스토리|컨셉|전체|도입|인트로|아웃트로|엔딩|보면|보니|보다가|볼\s*때)"
)
# 표지 뒤가 **공개 시점** 표현이면 뮤비는 때를 가리키는 말이지 내용이 아니다 — "뮤비 (처음) 나왔을 때 사람들이 다 따라 춤췄던 노래".
# 여는 말(처음·첫)보다 먼저 본다. "뮤비 처음 봤을 때 …"는 본 시점이라 여기 걸리지 않고 여는 말로 열린다.
_RELEASE_TIME_RE = re.compile(
    r"^\s*(?:가|는|도|이)?\s*(?:처음|막|딱|갓)?\s*"
    r"(?:나왔을\s*(?:때|무렵|당시)|나온\s*(?:지|뒤|후|날|직후|당시)|나오고\s*(?:나서|바로)|나오자마자|"
    r"공개(?:됐|된|되|했|할|\s*직후|\s*당시|\s*때)|발표(?:됐|된|되|했|\s*당시)|올라왔을\s*때|떴을\s*때|뜬\s*(?:지|뒤|후))"
)
# 조사 뒤 첫 절이 이렇게 끝나면 다음 절이 뮤비 내용이다 — "뮤비를 봤는데 …", "뮤비가 기억나는데 …". 부정이 없을 때만.
_CONTINUATION_RE = re.compile(
    r"봤|보니|보면|보다가|보는데|보고\s|기억나|기억이\s*나|기억에\s*남|생각나|떠오르|인상적|신기했|충격적|무서웠|이상했|독특했|특이했|기괴"
)
# 첫 절의 부정 — "안 찾아봤는데", "못 찾았는데", "안 보고", "있었는지 모르겠는데", "없어도". 뮤비 자체의 부정이다.
_CLAUSE_NEGATION_RE = re.compile(r"(?:안|못)\s*[가-힣]*(?:보|봤|찾|듣|들|알|기억)|않|없|모르|몰라")

# 표지에 바로 붙은 기억 부정 — 뮤비 자체를 모른다. "뮤비는 기억 안 나고", "뮤비 못 봤는데", "뮤비는 따로 없었어".
_MV_NEGATION_RE = re.compile(
    r"^\s*(?:는|도|를|은|가|이)?\s*(?:잘\s*|아예\s*|전혀\s*|따로\s*)?"
    r"(?:기억.{0,4}(?:안|않|못)|기억이?\s*없|모르|몰라|(?:안|못)\s*봤|본\s*적(?:이|은)?\s*없|생각.{0,4}(?:안|않)\s*나|없)"
)
# 장면 자체의 부정 — "장면은 못 봤어". '장면 말고는 기억 안 나'는 제외 문형이라 여기 걸리지 않는다.
_SCENE_NEGATION_RE = re.compile(
    r"(?:장면|씬)(?:은|는|도|을|이)?\s*(?:잘\s*)?(?:못\s*봤|기억.{0,4}(?:안|않|못)|모르|몰라|없)"
)

# 장면이라고 말한 것 / 화면 속성.
_SCENE_WORD_RE = re.compile(r"장면|씬(?![가-힣])|클로즈업")
_SCREEN_QUALITY_RE = re.compile(r"흑백|세피아|슬로\s*모션|역재생|거꾸로\s*재생")

# 눈에 보이는 동작·상태. 분위기 말(몽환적·웃기다·유명하다)은 넣지 않는다. 짧은 어간은 활용형으로 좁힌다.
_VISUAL_PREDICATE_RE = re.compile(
    r"떠\s*(?:있|다니|다녀|오르|올라|가|간)|떠있|떠다|뜬\s|뜨는|떴|날아|날리|떨어(?:지|뜨|진|져)|던지|던져|던진|매달|거꾸로|뒤집|"
    r"부서|부숴|부순|깨지|깨뜨|깨부|터지|터뜨|터진|폭발|타오르|불타|빠지|빠진|빠져|잠기|잠긴|젖은|젖어|"
    r"춤추|춤춰|춤췄|춤을\s*추|뛰어|달리(?:는|고|다|던)|달려(?!들)|걸어|걷는|걷고|서\s*있|서서|앉아|앉은|누워|누운|엎드|들고|안고|끌고|"
    r"입고|입은|쓰고|쓴\s|신고|타고(?!난)|탄\s|몰고|쏟아|쌓인|쌓여|흩날|흩어|흐르|쓰러|넘어지|기어|"
    r"올라가|올라서|내려가|내려오|내려(?!받)|올라오|뛰어내|점프|돌아다니|빙글|회전하|가라앉|비추|비친|비치는|밝히|"
    r"녹아|녹는|얼어|얼음|키스|포옹|껴안|안아|따라다니|따라다녀|따라오|따라가|쫓|공중|부양|내리는|오는\s|쏟아지|탈출|"
    r"숨어|숨는|달아나|도망|싸우|때리|울(?:고|어|었|던)|우는|웃(?:고|어|는|던)|불이\s*나|사라지|사라져|비(?:를)?\s*맞|"
    r"먹어|먹는|먹고|마시|노래(?:를)?\s*(?:불러|부르는|부른다)"
)
# 명사와 함께 장면을 이루는 구조어 — 배경·등장·존재. **구체 명사가 그 앞에 주어·장소로 올 때만** 장면이다
# ("교실이 나오는", "의자만 덩그러니 있어"). "뮤비가 유튜브에 있는 남자 아이돌"의 있는·"뮤비가 나온 지"의 나온은 아니다.
_SCENE_STRUCTURE_RE = re.compile(r"배경|나오|나와|나온|등장|보이|보여|보인|있(?:어|었|는|다|고)|나타|비춰|찍|촬영|가득|덩그러니")

# 가수 설명으로 쓴 남자/여자 — "남자 아이돌", "여자 솔로 곡"의 사람은 장면 속 인물이 아니다.
_ROLE_WORDS = r"(?!\s*(?:아이돌|솔로|가수|그룹|밴드|보컬|래퍼|발라드|댄스|힙합|알앤비|트로트|인디|노래|곡|목소리|버전|듀엣|듀오|가창|음색))"

# 아이·소년·소녀는 **낱말이 거기서 끝날 때만** 화면 속 인물이다(§2-29 ①, 2026-10-14). 명사 뒤에 올 수 있는 것은 닫힌 부류 둘뿐이다 —
# 조사(격조사·보조사·명사에 바로 붙는 '요', 넷까지 겹침: "아이들이랑" · "아이들만의" · "아이도요" · "소년들에게서는" · "아이들한테서도요")와
# 서술격 '이다'의 활용(평서 "소년이다" · "아이야" · "아이죠" · "소년이네요", 회상 "소녀였어" · "아이더라", 연결 "소년인데" · "아이들이고" ·
# "아이라서", 관형 "소녀인" · "아이였던", 의문 "소녀일까" · "아이냐", 양보 "아이거든" · "아이잖아").
# 그 뒤가 한글이 아니거나 문장 끝이거나 서술격이 시작되면 낱말이 끝난 것이다. 조사 뒤에 띄지 않고 시각 서술어·구조어를 붙여 쓴
# "아이가춤추는" · "소녀가울고" · "소년이나오는"도 낱말이 끝난 것으로 본다("교복입고" · "눈밭"을 지키는 것과 같은 태도 — PR #61 리뷰).
# 아이**돌** · 아이**유** · 아이**스크림** · 아이**라인** · 소녀**시대** · 소녀**감성** · 방탄소년**단**은 다른 낱말이다. 앞에 꾸미는 말이 붙은
# 어린**아이** · 청**소년** · 미**소년**은 그대로 인물이다.
_PERSON_PARTICLE = (
    # 격조사 — 주격·목적격·관형격·부사격(처소·여격·탈격·도구·공동)·호격
    r"(?:이|가|께서|에서|을|를|의|에|에게|한테|께|에게서|한테서|에게로|한테로|으로부터|로부터|으로서|로서|으로써|로써|으로|로|"
    r"와|과|랑|이랑|하고|보고|더러|아|야|"
    # 보조사 — 명사에 바로 붙는 '요' 포함
    r"은|는|도|만|뿐|까지|부터|마저|조차|밖에|마다|나|이나|든|이든|든지|이든지|라도|이라도|나마|이나마|야말로|이야말로|요|"
    r"처럼|같이|만큼|대로|보다|마냥|커녕|치고|씩|끼리)"
)
# '이'는 모음 뒤에서 흔히 준다("아이야" · "소녀였어" · "아이죠"). 홀로 선 '라'는 낱말 끝에서만("소년이라 그런지") — 아이**라인**은 아니다.
_PERSON_COPULA = (
    r"(?:이?(?:다|야|에요|예요|었|였|죠|지|네|군|구나|잖|거든|거야|더라|던|든|냐|고|며|면|니|"
    r"라서|라고|라는|란|라도|라니|라며|래|랍|라(?![가-힣]))|이자|인|일|입|임(?![가-힣]))"
)
_PERSON_WORD_END = (
    r"(?=" + _PERSON_PARTICLE + r"{0,4}(?:[^가-힣]|$|" + _PERSON_COPULA + r")"
    # 조사가 하나 이상 붙은 뒤 띄지 않고 바로 시각 서술어·구조어가 오면 낱말 끝이다("아이가춤추는"). 조사 없이 붙은 한글(아이**돌**)은 아니다
    r"|" + _PERSON_PARTICLE + r"{1,4}(?:" + _VISUAL_PREDICATE_RE.pattern + r"|" + _SCENE_STRUCTURE_RE.pattern + r"))"
)

# 구체 명사 — 사물·장소·자연·인물·동물. 눈에 보이는 것만. 짧은 동음이의어(길·방·배·말·눈·비·불·물·차·개·새)는
# 뒤에 오는 조사·서술어로 좁힌다. 신체 부위(얼굴)·추상(사랑·댓글·조회수)·직업명(배우·아이돌)은 넣지 않는다.
#
# 명사 앞뒤에 낱말 경계를 두지 않는다 — 앞에 꾸미는 말이 붙은 겹낱말(벚**꽃** · 밤**하늘** · 종이**비행기** · 어린**아이**)과 뒤에 붙는
# 말("꽃나무" · "교복입고" · "눈밭")이 많다. 그래서 명사를 품은 **가수 이름**(잔**나비** · 유리**상자** · 임한**별** · 태양 · 별 · 버스커 버스커)도
# 명사로 읽힌다 — "춤추는 잔나비 노래"와 "흩날리는 벚꽃 노래"는 문형이 같아 문장만으로는 가를 수 없다(§2-29 ③, 10/14에 낱말 시작
# 경계·노래 앞 자리 규칙을 시도했으나 실제 장면을 잃어 보류). 가수 이름을 가리려면 이름 사전(§2-29 ④·⑤)이 필요하다.
_CONCRETE_NOUN_RE = re.compile(
    r"의자|책상|소파|침대|거울|창문|문짝|우산|풍선|꽃(?:잎|다발|밭)?|나무|촛불|전화기|카메라|시계|피아노|기타(?=를|가|\s*치)|드럼|마이크|"
    r"인형|가면|칼|총|편지|사진기|TV|티비|텔레비전|자전거|오토바이|자동차|차(?=[가를에]\s)|버스|기차|지하철|트럭|택시|비행기|로켓|우주선|"
    r"상자|의상|교복|드레스|한복|정장|모자|신발|구두|가방|케이크|수박|사과|컵라면|풍차|그네|회전목마|계단|엘리베이터|"
    r"하늘|구름|달(?=이|을|빛|\s)|별(?=이|을|빛|들|\s)|태양|바다|해변|파도|강(?=가|물|에\s)|호수|수영장|물속|물(?=\s*속|에\s*(?:빠|잠|뛰|들어)|\s*위)|"
    r"비(?=가\s*(?:계속\s*|주룩주룩\s*|쏟아)?(?:내|오|쏟)|\s*오는|\s*내리|가\s*와|\s*맞|를\s*맞)|눈(?=\s*(?:오는|내리)|이\s*(?:오|내)|밭|보라|사람)|불(?=이\s*(?:타|붙|나)|을\s*(?:지르|붙)|길|꽃|바다|이\s*번)|"
    r"옥상|지붕|건물|빌딩|골목|도로|길거리|거리(?=에서|를|가)|교실|학교|병원|공항|사막|들판|숲|설원|터널|다리(?=\s*위|에서|를)|놀이공원|편의점|"
    r"지하실|창고|공장|주차장|도시|시골|우주|밤거리|복도|천장|바닥|벽(?=에|을|이)|방(?=\s*안|에서|\s*안에)|감옥|교회|성당|무덤|묘지|"
    r"남자" + _ROLE_WORDS + r"|여자" + _ROLE_WORDS + r"|(?:소년|소녀|아이)들?" + _PERSON_WORD_END + r"|"
    r"아기|주인공|사람들?|멤버들?|무용수|댄서|로봇|외계인|경찰|군인|신부|천사|악마|유령|괴물|"
    r"고양이|강아지|개(?=[가를]\s)|새(?=[가를]\s|들|떼)|말(?=이\s*(?:달|뛰)|을\s*타)|사자|호랑이|나비|물고기|고래|상어|뱀|쥐|토끼|곰|사슴|늑대"
)

# 구체 명사 + 구조어("교실이 나오는"). 시각 서술어·구조어 자체는 위(`_VISUAL_PREDICATE_RE` · `_SCENE_STRUCTURE_RE`)에 있다.
_NOUN_THEN_STRUCTURE_RE = re.compile(
    r"(?:" + _CONCRETE_NOUN_RE.pattern + r")(?:[가-힣]{0,2})?\s*(?:\S+\s+){0,2}(?:" + _SCENE_STRUCTURE_RE.pattern + r")"
)
# "뮤비 배경이 사막이야" — 배경 뒤에 구체 명사.
_BACKGROUND_THEN_NOUN_RE = re.compile(r"배경(?:이|은|는|으로|을)?\s*(?:\S+\s+){0,2}(?:" + _CONCRETE_NOUN_RE.pattern + r")")

# 시각 서술어와 겹치는 관용구 — 장면이 아니다.
_IDIOM_RE = re.compile(r"사랑에\s*빠|잠에\s*빠|생각에\s*잠|추억에\s*잠|눈에\s*띄|마음에\s*들|시간이\s*흐르|세월이\s*흐르")

# 장면이 불확실하다 — 불확실 어미가 **시각 서술어 어절 자체**에 붙었을 때만("떠 있었는지는 기억이 안 나", "떠 있었나 잘 모르겠어",
# "뛰어내렸나 그랬는데").
# ", 누가 불렀는지는 모르겠어"처럼 다른 대상의 불확실은 쉼표·절을 넘어 장면을 지우지 않는다.
_SCENE_UNCERTAIN_RE = re.compile(
    r"(?:" + _VISUAL_PREDICATE_RE.pattern + r")[가-힣]{0,3}(?:는지|은지|나)\s*(?:는|도|가|를)?\s*(?:잘\s*|정확히\s*|제대로\s*)?(?:기억|모르|몰라|그랬|그런|싶|같)"
)


def _strip_quoted(text: str) -> str:
    return _MENTIONED_AS_WORD_RE.sub(" ", _QUOTED_RE.sub(" ", text))


_DECLARATIVE_THEN_TOPIC_RE = re.compile(r"(?:어|아|야|여|해|러|워|요)(?:는|은)$")


def _ends_attributively(text: str) -> bool:
    """마지막 어절이 관형형(…는/…은/…던/…ㄴ/…ㄹ)으로 끝나는가."""
    text = text.rstrip()
    if not text or text.endswith(","):
        return False
    # 종결형 뒤의 는/은은 주제 조사다("노래 불러는", "내용이야는") — 관형형이 아니다.
    if _DECLARATIVE_THEN_TOPIC_RE.search(text):
        return False
    last = text[-1]
    if last in "는은던을":
        return True
    code = ord(last) - 0xAC00
    if 0 <= code < 11172:
        return code % 28 in (4, 8)   # 받침 ㄴ·ㄹ (뜬·던진·떠있는·날)
    return False


def _before_span(before: str) -> str:
    """표지 앞 장면 구간 — 표지 바로 앞 어절이 **시각 서술어·구조어·화면 속성의 관형형**일 때만("떠 있는 뮤비",
    "교실이 나오는 뮤비"). "발라드는 뮤비가…"의 '는'은 주제 조사라 관형절이 아니다. 앞 절(쉼표·연결어미로 끊김)은 다른 화제다."""
    if not _ends_attributively(before):
        return ""
    tail = " ".join(before.split()[-2:])
    if not (_VISUAL_PREDICATE_RE.search(tail) or _SCENE_STRUCTURE_RE.search(tail) or _SCREEN_QUALITY_RE.search(tail)):
        return ""
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(before))
    return before[boundaries[-1].end():] if boundaries else before


def _first_clause(after: str) -> str:
    """첫 절. 경계 어절("봤는데", "보면")까지 포함한다 — 이어지는 말 판정이 그 어절을 봐야 한다."""
    cut = _CLAUSE_BOUNDARY_RE.search(after)
    return after[: cut.end()] if cut else after


def _cut_topic(after: str) -> str:
    cut = _TOPIC_SWITCH_RE.search(after)
    return after[: cut.start()] if cut else after


def _describes_scene(span: str) -> bool:
    """이 구간이 눈에 보이는 장면을 말하는가."""
    if _SCENE_WORD_RE.search(span) or _SCREEN_QUALITY_RE.search(span):
        return True
    cleaned = _IDIOM_RE.sub(" ", span)
    if _CONCRETE_NOUN_RE.search(span) and _VISUAL_PREDICATE_RE.search(cleaned):
        return True
    return bool(_NOUN_THEN_STRUCTURE_RE.search(cleaned) or _BACKGROUND_THEN_NOUN_RE.search(cleaned))


def _after_span(after: str) -> str:
    """표지 뒤 장면 구간. 빈 문자열이면 이 표지는 장면을 열지 않았다.

    여는 말이면 뒤 전부. 조사 뒤 첫 절에 부정이 있으면 뮤비 자체의 부정(없음). 첫 절이 장면을 담거나
    이어지는 말이면 뒤 전부. 그 외는 첫 절까지만 — "뮤비가 유명해서 …", "뮤비도 화제였고 …"의 뒤 절은 뮤비 내용이 아니다.
    """
    after = _cut_topic(after)
    if _RELEASE_TIME_RE.search(after):
        return ""   # 공개 시점 부사구 — 뒤 절은 유행·반응 이야기다
    if _OPENER_RE.search(after):
        return after
    head = _first_clause(after)
    if _CLAUSE_NEGATION_RE.search(head):
        return ""
    if _describes_scene(head) or _CONTINUATION_RE.search(head):
        return after
    return head


def is_specific_mv_scene_query(query: str) -> bool:
    """사용자가 **뮤직비디오 속 구체적인 장면**을 단서로 말했는가. 원문만 본다.

    지원 범위(`tests/test_rerank_policy.py`로 고정 — 예문 목록과 성질 시험):
      대상   — "뮤비에 의자가 하늘에 떠있는 장면이 있어", "뮤비를 봤는데 의자가 하늘에 떠 있었어",
               "뮤비에서 교복 입고 춤추는 노래", "뮤비 배경이 사막이야", "뮤비가 흑백이고 비가 계속 내려",
               "제목은 기억 안 나는데 뮤비에서 의자가 하늘에 떠 있었어"(다른 대상의 부정은 무시),
               "뮤직비디오에서 의자를 던져, 누가 불렀는지는 모르겠어"(다른 대상의 불확실은 무시),
               "뮤비에서 의자를 던지는 장면 말고는 기억 안 나"(부정이 장면을 제외하고 말함)
      비대상 — "뮤비가 유명한 노래"·"뮤비 조회수가 계속 올라가던"(장면 없음), "뮤직비디오가 코미디처럼 웃겼고"(감성),
               "남자 그룹이고 뮤비가 유튜브에 있어"·"여자 발라드는 뮤비가 예뻤어"·"뮤비가 유튜브에 있는 남자 아이돌 노래"
               (뮤비 밖의 설명, 가수 설명의 남자/여자), "뮤비는 안 찾아봤는데 …"(부정 뒤는 열지 않음),
               "의자가 떠 있는 것 같은 몽환적인 분위기"(뮤비 언급 없음), "가사에 '뮤비에서 의자를 던져'라는 말"(인용),
               "뮤비는 기억 안 나고 남자 발라드야"·"뮤비 못 봤는데"(뮤비 자체를 모름),
               "뮤비에서 의자가 떠 있었는지는 기억이 안 나"(장면이 불확실),
               "뮤비에 배우 김태리가 나와"(출연 사실 — 구체 명사가 없어 범위 밖. 인물 출연은 장면이 아니라 사실 주장이다),
               "뮤비에서 춤추는 아이돌/소녀시대/방탄소년단 노래"·"뮤비에서 춤추는 여자 목소리 노래"
               (가수 설명 — 아이·소녀·소년은 낱말이 거기서 끝날 때만(조사·서술격 활용), 남자/여자+목소리·음색·보컬은 가수 화제.
               2026-10-14, §2-29 ①·②)
      한계   — 명사 목록 밖의 사물이 구조어와만 쓰이면 비대상("뮤비에 트램펄린이 나와"). 표지 앞 관형절은 마지막
               절 경계 뒤부터라 "교복 입고 춤추는 뮤비"는 '교복'이 잘린다. "뮤비는 기억이 잘 안 나지만 의자가 떠 있었던
               건 확실해"처럼 뮤비 부정 뒤에 장면을 확신하는 문형은 비대상이다. 공개 시점 부사구 뒤에 장면을 이어 말한
               "뮤비 나왔을 때 봤는데 의자가 하늘에 떠 있었어"도 비대상이다 — 시점 부사구 뒤를 다시 열면 "뮤비 나왔을 때
               봤는데 사람들이 다 따라 춤췄어"의 유행 이야기가 장면으로 읽혀, 비우는 쪽을 택했다. 놓치면 현행 동작(리랭킹)이다.
               구체 명사를 품은 가수 이름은 명사로 읽혀 대상이다 — "뮤비에서 춤추는 잔나비/유리상자/임한별/태양/별/버스커 버스커 노래",
               "뮤비에서 잔나비가 춤추는 노래"(§2-29 ③·④·⑤, 미착수). "흩날리는 벚꽃 노래"·"날아다니는 종이비행기 노래"와 문형이
               같아 문장만으로는 가를 수 없고, 가수 이름 사전이 있어야 한다. 이름이 인물 낱말로 끝나는 아이오아이도 같다.

    분석 프롬프트의 산출물(has_visual_clue 등)을 쓰지 않는다. 그래야 규칙 폴백 분석과
    prior_analysis 재검색에서도 같은 판정이 나온다.
    """
    return bool(mv_scene_spans(query))


def mv_scene_spans(query: str) -> List[str]:
    """`is_specific_mv_scene_query`가 장면으로 판정한 **구간의 원문**(공백 정리). 판정 규칙은 같다 — 이 함수가
    비어 있지 않은 것과 그 함수가 True인 것은 동치다.

    사용자용 설명이 "말씀하신 뮤직비디오 장면"을 사용자의 말 그대로 보여 주기 위한 것이다. 판정에는 쓰지
    않는다(판정은 bool 하나로 끝난다). 생략 정책 자체는 바꾸지 않는다.
    """
    text = _strip_quoted(query or "")
    spans: List[str] = []
    if not _MV_MARKER_RE.search(text):
        return spans
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for marker in _MV_MARKER_RE.finditer(sentence):
            after_raw = sentence[marker.end():]
            if _MV_NEGATION_RE.search(after_raw):
                continue   # 뮤비 자체를 모른다
            before = _before_span(sentence[: marker.start()])
            after = _after_span(after_raw)
            span = before + marker.group(0) + after
            if _SCENE_UNCERTAIN_RE.search(span) or _SCENE_NEGATION_RE.search(span):
                continue
            if _describes_scene(before) or _describes_scene(after):
                spans.append(" ".join(span.split()))
    return spans


# ---------------------------------------------------------------------------
# 설명 표시용 장면 구간 — 판정에는 쓰지 않는다
# ---------------------------------------------------------------------------
#
# 판정 구간(`mv_scene_spans`)은 여는 말 뒤 문장 끝까지라 장면이 아닌 말이 딸려 온다 — 찾는 곡을 가리키는 말
# ("… 떠 있는 그 노래 뭐지", "… 걷는 여자 가수 노래야")과 다른 단서의 절("…, 누가 불렀는지는 모르겠어").
# 사용자에게 "말씀하신 장면"으로 보일 칸에는 장면을 말한 부분만 남긴다. 자른 결과가 장면이 아니면 자르지 않는다 —
# 길 뿐 틀리지는 않은 쪽을 택한다.

# 찾는 곡을 가리키는 말의 낱말들(어절 전체가 맞아야 한다 — "남자가"·"곡선"은 아니다).
_REF_DEMONSTRATIVE_RE = re.compile(r"그런|이런|저런|그|이|저")
_REF_GENDER_RE = re.compile(r"남자|여자|남성|여성|혼성")
# 곡을 꾸미는 말 — 가창자 형태와 장르. "여자 가수 노래", "남자 아이돌 곡", "발라드 노래".
_REF_ROLE = (r"(?:보이그룹|걸그룹|아이돌|가수|그룹|밴드|솔로|래퍼|듀오|듀엣|보컬|"
             r"발라드|댄스|힙합|랩|록|락|메탈|알앤비|R&B|인디|포크|트로트|재즈|일렉|일렉트로닉|어쿠스틱|팝|케이팝|K-?POP|OST)")
_REF_ROLE_RE = re.compile(_REF_ROLE, re.IGNORECASE)
_REF_COPULA = r"(?:이?야|였[가-힣]{0,3}|이었[가-힣]{0,3}|인데|이?에요|예요|입니다|임)"
_REF_SONG_RE = re.compile(r"(?:노래|곡)(?:들)?(?:" + _REF_COPULA + r"|요|를|을|가|이|는|은|도|만|좀)?|노랜데?")
# 지시어 없는 "노래/곡" 바로 뒤가 이러면 장면 속에서 부르고 듣는 노래다 — "노래 부르면서 걷는", "노래를 틀어 놓고".
_SONG_AS_OBJECT_RE = re.compile(r"\s*(?:부르|불러|불렀|부른|하(?:는|고|면서|며|다가)|틀|듣|들으|나오|나와|흘러|따라)")


# 화면 속에서 노래하는 행위("비 맞으면서 노래 부르고 있어").
_SINGING_RE = re.compile(r"노래(?:를)?\s*(?:부르|불러|불렀|부른)")
_WORD_RE = re.compile(r"\S+")
# "배경이 사막" — 배경 뒤의 **가장 가까운** 명사까지. 판정용 `_BACKGROUND_THEN_NOUN_RE`는 두 어절 안의 먼 명사도
# 잡으므로("배경이 사막인 여자 (목소리)") 표시 구간에서는 이 판을 쓴다. 판정에는 쓰지 않는다.
_BACKGROUND_THEN_NEAREST_NOUN_RE = re.compile(
    r"배경(?:이|은|는|으로|을)?\s*(?:\S+\s+){0,2}?(?:" + _CONCRETE_NOUN_RE.pattern + r")"
)
# 보조 용언 · 구조어 — **어간에 어미가 붙은 낱말**일 때만(어절 전체가 맞아야 한다). 판정용 `_SCENE_STRUCTURE_RE`는
# 낱말의 앞부분만 봐도 맞으므로 "보이그룹"·"배경음악"·"찍먹"·"있잖아"까지 읽는다 — 표시 구간에서는 쓰지 않는다.
_AUX_WORD_RE = re.compile(
    r"(?:있|계시)(?:어|어요|어서|었[가-힣]{0,3}|는|는데|는지|다|다가|고|던|음|습니다|더라|네|지|지만)|계셔[가-힣]{0,2}"
)
_STRUCTURE_WORD_RE = re.compile(
    r"(?:나오|나타나|등장하|보이|비추|찍히|가득하)(?:는|는데|고|던|네|더라|며|면서|지|지만)"
    r"|나와(?:요|서)?|나왔[가-힣]{0,3}|나온(?:다)?|나타났[가-힣]{0,3}|나타난(?:다)?"
    r"|등장(?:해|해요|했[가-힣]{0,3}|한|한다)?|보여(?:요|서)?|보였[가-힣]{0,3}|보인(?:다)?"
    r"|비춰(?:요|서)?|비췄[가-힣]{0,3}|비춘(?:다)?|찍(?:어|어요|었[가-힣]{0,3}|은|는|고)|찍혀(?:요|서)?|찍혔[가-힣]{0,3}|찍힌"
    r"|촬영(?:해|했[가-힣]{0,3}|한|된|됐[가-힣]{0,3}|하는)?|가득(?:해|해요|했[가-힣]{0,3}|한|하고)?|덩그러니"
    r"|배경(?:이|은|는|으로|을|인|이야|이에요|이었[가-힣]{0,3}|임|이고)?"
)

# 장면 서술이 어떻게 끝났는가 — 그 뒤를 이어 붙일지 정한다.
_FINAL = "final"              # 종결어미·명사 서술로 닫혔다("떠 있어", "사막이야", "여자") — 뒤는 잇지 않는다
_ATTRIBUTIVE = "attributive"  # 관형형으로 닫혔다("떠 있는") — 뒤의 말이 혼자 장면일 때만 잇는다
_CONNECTIVE = "connective"    # 그 밖의 연결어미("있는데", "앉아서") — 관형형과 같다
_LISTING = "listing"          # 나열 연결어미로 열려 있다("입고", "맞으면서") — 같은 절의 다음 서술을 잇는다
_ARGUMENT = "argument"        # 조사가 붙은 명사("차가", "장면이") — 서술이 아직 오지 않았다. 나열과 같다
_LISTING_END_RE = re.compile(r"(?:고|며|면서)$")
_CONNECTIVE_END_RE = re.compile(r"(?:는데|은데|인데|지만|아서|어서|해서|여서|다가|더니|니까|으면|자마자)$")
_NOMINAL_FINAL_RE = re.compile(r"(?:들)?(?:이?야|였[가-힣]{0,3}|이었[가-힣]{0,3}|이?에요|예요|입니다|임)?")   # 맨명사 포함
_TRAILING_PUNCTUATION = ",.!?~"
# 나열·명사 뒤에서 다음 서술까지 사이에 낄 수 있는 말 — 가리키는 말과 부사. 모르는 낱말이 끼면 잇지 않는다
# (그것이 앞 장면을 닫은 서술어일 수 있다 — "기차 타고 | 떠나 | 춤추기 좋은").
_BETWEEN_WORD_RE = re.compile(
    r"그|이|저|그걸|그게|그건|이걸|이게|그것을|그것이|계속|계속해서|막|마구|다|다같이|같이|함께|혼자|갑자기|진짜|엄청|되게|완전|"
    r"또|다시|서로|둥둥|빙글빙글|마지막에|처음에|중간에|끝에|[가-힣]+(?:게|히)"
)
_SONG_OBJECT_RE = re.compile(r"(?:노래|곡)(?:를|을)?")
# "-아/어 있다"로 상태를 말하는 용언 — 이 뒤의 "있어"는 그 서술의 일부다("앉아 있어", "매달려 있어").
# 동작을 끝맺은 말 뒤의 "있어"는 아니다("춤춰 | 있어 보이는 노래야").
_RESULTATIVE_WORD_RE = re.compile(
    r"앉아|누워|엎드려|매달려|걸려|쌓여|젖어|숨어|놓여|붙어|서|떠|잠겨|묻혀|갇혀|쓰러져|넘어져|떨어져|깨져|부서져|꽂혀|덮여|가라앉아"
)
# 서술어 목록에 있지만 용언이 아닌 낱말 — 부사("거꾸로 매달린")와 '하다'가 붙지 않은 명사("키스 장면"). 서술이 닫히지 않았다.
_OPEN_PREDICATE_WORD_RE = re.compile(
    r"거꾸로|빙글(?:빙글)?|덩그러니|가득|(?:공중|부양|얼음|키스|포옹|점프|폭발|탈출|도망)(?:을|를|이|가|은|는|도|만|에|에서|으로)?"
)
# 화면 속 사람을 가리키는 성별 낱말(뒤에 가수 형태·장르·목소리가 오지 않는다). 판정의 `_CONCRETE_NOUN_RE`와 같은 조건이다.
_GENDER_PERSON_RE = re.compile(r"(?:남자|여자)" + _ROLE_WORDS)
_GENDER_WORD_RE = re.compile(r"남자|여자")


def _mask_singer_genders(text: str, sentence: str) -> str:
    """**문장 전체에서** 가수를 가리키는 성별 낱말("남자 가수", "여자 목소리")을 글자 수를 지켜 가린다.

    앞부분만 잘라 보면 "걷는 남자 | 가수 곡"의 남자가, 판정이 화제 전환어 앞에서 자른 구간에서는 "떠 있는 여자 |
    목소리"의 여자가 화면 속 사람으로 읽힌다(뒤 낱말이 보이지 않아서). 표시 구간을 정할 때만 쓴다.
    """
    chars = list(text)
    for match in _GENDER_WORD_RE.finditer(text):
        if not _GENDER_PERSON_RE.match(sentence, match.start()):
            chars[match.start(): match.end()] = "□" * (match.end() - match.start())
    return "".join(chars)


class _SceneWords:
    """표시 구간을 정하려고 한 번 읽어 둔 것 — 어절, 명사 어절, 서술 어절, 장면을 서술한 어절(anchor).

    명사는 앞뒤 낱말을 보고 정해지므로("남자 가수"의 남자는 아니다) 어절을 떼어 보지 않고 전체에서 찾는다.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.words = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
        cleaned = _IDIOM_RE.sub(lambda m: " " * len(m.group()), text)      # 자리를 보존한 채 관용구만 지운다
        # 명사·'장면' 어절 → 그 낱말이 끝나는 자리(어미·조사가 시작하는 곳)
        self.nouns = {}
        for rx in (_CONCRETE_NOUN_RE, _SCENE_WORD_RE):
            for m in rx.finditer(text):
                self.nouns.setdefault(self._word_at(m.start()), m.end())
        self.nouns.pop(-1, None)
        self.predicates = set()
        self.inside_predicate = set()     # 문형 하나가 걸친 어절 전부 — "떠 / 있어"의 떠, "노래 / 불러"의 노래
        for rx, hay in ((_SCREEN_QUALITY_RE, text), (_SINGING_RE, text), (_VISUAL_PREDICATE_RE, cleaned)):
            for m in rx.finditer(hay):
                last = self._last_word_of(m)
                self.predicates.add(last)
                self.inside_predicate.update(range(self._word_at(m.start()), last + 1))
        self.predicates.discard(-1)
        self.anchors = set(self.predicates)
        self.anchors.update(i for i in self.nouns if _SCENE_WORD_RE.search(self.words[i][2]))
        for m in _BACKGROUND_THEN_NEAREST_NOUN_RE.finditer(cleaned):
            last = self._last_word_of(m)
            self.anchors.add(last)
            self.inside_predicate.update(range(self._word_at(m.start()), last + 1))     # "배경이 / 사막이야"의 배경이
        self.anchors.discard(-1)
        self._license_structure_words()
        self._add_nominal_predicates()

    def _word_at(self, position: int) -> int:
        for index, (start, end, _) in enumerate(self.words):
            if start <= position < end:
                return index
        return -1

    def _last_word_of(self, match: "re.Match[str]") -> int:
        end = match.end()
        while end > match.start() and self.text[end - 1].isspace():       # 일부 문형은 뒤 공백까지 문다("오는 ")
            end -= 1
        return self._word_at(end - 1)

    def bare(self, index: int) -> str:
        return self.words[index][2].rstrip(_TRAILING_PUNCTUATION)

    def is_support(self, index: int) -> bool:
        word = self.bare(index)
        return bool(_STRUCTURE_WORD_RE.fullmatch(word) or _AUX_WORD_RE.fullmatch(word))

    def _license_structure_words(self) -> None:
        """구체 명사에 붙은 구조어 — "교실이 나와", "의자만 덩그러니", "눈이 내리고 있어". 사이에 낀 말은 둘까지.
        서술어·구조어를 넘어가지 않는다(이미 서술된 명사다)."""
        for index in range(len(self.words)):
            if not self.is_support(index):
                continue
            back, skipped = index - 1, 0
            while back >= 0:
                if back in self.nouns:
                    self.anchors.add(index)
                    break
                if back in self.predicates or self.is_support(back) or skipped >= 2:
                    break
                skipped += 1
                back -= 1

    def _add_nominal_predicates(self) -> None:
        """주어가 있는 명사 서술 — "사람은 여자", "주인공은 여자였어", "마지막 장면은 바다였어".
        조사가 붙은 명사("남자가")와 서술 없이 놓인 명사("떠 있는 | 남자")는 해당하지 않는다."""
        for index, noun_end in self.nouns.items():
            if index <= 0 or index - 1 not in self.nouns:
                continue
            if not _NOMINAL_FINAL_RE.fullmatch(self.text[noun_end: self.words[index][0] + len(self.bare(index))]):
                continue
            if self.bare(index - 1)[-1] in "은는이가":
                self.anchors.add(index)

    def closure(self, index: int) -> str:
        """이 어절에서 서술이 어떻게 끝났는가."""
        word = self.bare(index)
        if index in self.nouns and index not in self.predicates:
            ending = self.text[self.nouns[index]: self.words[index][0] + len(word)]
            if _NOMINAL_FINAL_RE.fullmatch(ending):
                return _FINAL                     # "사막이야", "여자였어", "장면"
            if ending in ("인", "이던", "이었던"):
                return _ATTRIBUTIVE               # "사막인"
            if ending in ("이고", "이며"):
                return _LISTING
            if ending.endswith("인데"):
                return _CONNECTIVE
            return _ARGUMENT                      # "차가", "장면이", "의자를"
        if _LISTING_END_RE.search(word) or _OPEN_PREDICATE_WORD_RE.fullmatch(word):
            return _LISTING                       # "입고", "맞으면서" · "거꾸로 (매달린)"
        if _CONNECTIVE_END_RE.search(word):
            return _CONNECTIVE
        if _ends_attributively(word):
            return _ATTRIBUTIVE
        return _FINAL

    def with_support(self, index: int) -> int:
        """바로 뒤에 붙어 이 서술을 끝맺는 보조 용언까지 — 열린 서술 뒤("부르고 / 있어", "장면이 / 있는데")와
        상태를 말하는 용언 뒤("앉아 / 있어"). 닫힌 서술 뒤에는 붙이지 않는다("떠 있는 | 있는", "춤춰 | 있어 보이는")."""
        while index + 1 < len(self.words):
            kind = self.closure(index)
            following = self.bare(index + 1)
            open_ended = kind in (_LISTING, _ARGUMENT)                    # "부르고 / 있어", "장면이 / 있는데"
            takes_aux = open_ended or (kind == _FINAL and _RESULTATIVE_WORD_RE.fullmatch(self.bare(index)))
            if takes_aux and _AUX_WORD_RE.fullmatch(following):
                index += 1
                continue
            break
        return index

    def next_scene_word(self, after_index: int) -> int:
        """이어지는 다음 서술. 없거나, 멀거나, 모르는 낱말이 끼면 -1 — 사이에는 명사·가리키는 말·부사만 올 수 있다."""
        following = min((i for i in self.anchors if i > after_index), default=-1)
        if following < 0 or following - after_index > 4:
            return -1
        for index in range(after_index + 1, following):
            word = self.bare(index)
            modifies_argument = _ends_attributively(word) and (
                index + 1 in self.nouns or _SONG_OBJECT_RE.fullmatch(self.bare(index + 1))
            )
            if not (index in self.nouns or index in self.inside_predicate or _BETWEEN_WORD_RE.fullmatch(word)
                    or _SONG_OBJECT_RE.fullmatch(word) or modifies_argument):
                return -1
        return following


def _starts_with_song_reference(text: str) -> bool:
    """닫힌 장면 뒤의 이 말이 **찾는 곡을 가리키는 말**로 시작하는가 — [지시어] [성별] [가수 형태·장르…] 노래/곡.

    "떠 있는 | 그 노래", "걷는 | 여자 가수 노래", "빠지는데 | 그 노래". 지시어 없이 "노래 부르는"처럼 장면 속
    행위의 대상이면 장면 안의 노래다. 지시어가 붙은 노래는 뒤에 무엇이 와도 찾는 곡으로 본다 — 짧게 자르는 쪽이다.
    """
    words = [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
    index = 0
    has_demonstrative = bool(words) and bool(_REF_DEMONSTRATIVE_RE.fullmatch(words[0][2]))
    if has_demonstrative:
        index += 1
    if index < len(words) and _REF_GENDER_RE.fullmatch(words[index][2]):
        index += 1
    while index < len(words) and _REF_ROLE_RE.fullmatch(words[index][2]):
        index += 1
    if index >= len(words) or not _REF_SONG_RE.fullmatch(words[index][2]):
        return False
    # 지시어 없이 부르고 듣는 노래는 장면 속 행위의 대상이다("걷는 | 노래 부르는 여자가 나와")
    return has_demonstrative or not _SONG_AS_OBJECT_RE.match(text[words[index][1]:])


def _scene_part(text: str, sentence: str) -> str:
    """`text`(표지 뒤 판정 구간)에서 **장면을 말한 앞부분**. 장면이 성립하지 않으면 빈 문자열.

    `sentence`는 같은 자리에서 시작하는 문장의 나머지다(판정이 화제 전환어 앞에서 자르기 전). 잘린 조각만 보면
    "춤추는 여자 | 목소리"의 여자가 화면 속 사람으로 읽히므로, 문장 전체에서 가수를 가리키는 성별 낱말은 가리고 본다.

    1. 장면이 성립하는 **가장 짧은** 앞부분에서 시작한다(판정과 같은 `_describes_scene`).
    2. 그 서술이 어떻게 끝났는지에 따라 잇는다.
       - 종결어미·명사 서술로 닫혔다 → 끝. 뒤는 다른 말이다("… 떠 있어 | 울고 싶을 때 들었어", "| 부른 사람은 여자야").
       - 나열 연결어미(-고·-며·-면서)·조사 붙은 명사로 열려 있다 → 같은 절의 다음 서술까지("교복 입고 / 춤추는",
         "흑백이고 / 비가 계속 내려", "차가 / 폭발해"). 모르는 낱말이 끼면 잇지 않는다.
       - 관형형·그 밖의 연결어미로 닫혔다 → 뒤의 말이 **혼자 장면일 때만**, 그 장면 부분을 잇는다
         ("서 있는 / 여자가 나오는", "떠있는 / 장면이 있어"). 곡을 가리키는 말로 시작하면 잇지 않는다.
    판단이 서지 않으면 짧게 자른다 — 덜 끝난 칸은 틀리지 않지만, 말하지 않은 것이 든 칸은 틀린다.
    """
    seen = _mask_singer_genders(text, sentence)        # 가수를 가리키는 성별 낱말을 가린 판 — 자리는 text와 같다
    info = _SceneWords(seen)
    words = info.words
    first = next((i for i in range(len(words)) if _describes_scene(seen[: words[i][1]] + " ")), -1)
    if first < 0:
        return ""
    last = info.with_support(first)
    while True:
        end = words[last][1]
        kind = info.closure(last)
        if kind == _FINAL or not text[end:].strip():
            break
        if kind in (_LISTING, _ARGUMENT):
            following = info.next_scene_word(last)
            if following < 0:
                break
            last = info.with_support(following)
            continue
        if _starts_with_song_reference(text[end:]):
            break               # 찾는 곡을 가리키는 말이다 — 그 뒤는 곡 이야기다
        return text[:end] + _scene_part(text[end:], sentence[end:])   # 뒤의 말이 혼자 장면이 아니면 빈 문자열이다
    return text[: words[last][1]]


def _tidy(phrase: str) -> str:
    return " ".join(phrase.split()).rstrip(" ,·")


def mv_scene_phrases(query: str) -> List[str]:
    """**설명 표시용** 장면 구간. 비어 있지 않은 것은 `is_specific_mv_scene_query`가 True인 것과 동치다.

    `mv_scene_spans`와 같은 판정에서 출발해 표지마다 구간을 줄인다.
      1. 다음 뮤비 표지 앞에서 자른다 — 표지가 둘이면 앞 구간이 뒤 구간을 품지 않게.
      2. 장면을 말한 앞부분만 남긴다(`_scene_part`) — 장면이 성립하는 가장 짧은 앞부분에서 시작해, 그 서술이
         열려 있을 때만 잇는다. 종결어미로 닫힌 뒤의 말("… 떠 있어 남자가 불렀어")은 싣지 않는다.
    줄인 결과가 장면이 아니면(`_describes_scene`) 줄이기 전 구간을 쓴다. 판정에는 쓰지 않는다 — 정책은 그대로다.
    """
    text = _strip_quoted(query or "")
    if not _MV_MARKER_RE.search(text):
        return []
    phrases: List[str] = []
    judged = False
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        markers = list(_MV_MARKER_RE.finditer(sentence))
        for index, marker in enumerate(markers):
            after_raw = sentence[marker.end():]
            if _MV_NEGATION_RE.search(after_raw):
                continue
            before = _before_span(sentence[: marker.start()])
            after = _after_span(after_raw)
            span = before + marker.group(0) + after
            if _SCENE_UNCERTAIN_RE.search(span) or _SCENE_NEGATION_RE.search(span):
                continue
            before_scene = _describes_scene(before)
            if not (before_scene or _describes_scene(after)):
                continue
            judged = True
            # 1. 다음 표지 앞까지만 — 그 뒤는 다음 표지의 구간이다
            own = after
            if index + 1 < len(markers):
                own = after[: max(0, markers[index + 1].start() - marker.end())]
            # 2·3. 장면을 말한 부분만. 표지 뒤가 장면이 아니면(장면은 표지 앞 관형절에 있다) 뒤는 싣지 않는다.
            # 줄여서 장면이 아니게 되면 줄이지 않는다
            own_scene = _describes_scene(own)
            if not (before_scene or own_scene):
                continue    # 이 표지의 장면은 다음 표지 뒤에 있다 — 그 표지가 보인다
            kept = _scene_part(own, after_raw)
            if own_scene and not kept:
                kept = own      # 판정은 장면으로 봤는데 줄일 수 없다("춤추는 여자 | 목소리 노래") — 판정 구간 그대로
            phrase = _tidy(before + marker.group(0) + kept)
            if phrase:
                phrases.append(phrase)
    if judged and not phrases:
        phrases = [_tidy(span) for span in mv_scene_spans(query)]   # 줄일 수 없었다 — 판정 구간 그대로
    # 겹치는 구간은 한 번만 — 다른 구간에 통째로 들어 있는 것은 뺀다
    unique: List[str] = []
    for phrase in phrases:
        if phrase and not any(phrase != other and phrase in other for other in phrases) and phrase not in unique:
            unique.append(phrase)
    return unique


# ---------------------------------------------------------------------------
# 장면을 판단할 출처 자료가 있는가
# ---------------------------------------------------------------------------

# 후보(MatchingTrack)에서 **뮤직비디오 장면의 출처 자료**로 인정하는 필드. 지금은 없다.
#
# 다음은 자료가 **아니다** — 같은 낱말이 있어도 장면의 증거로 승격하지 않는다.
#   lyrics_summary / lyrics_highlight : 가사 속 장면(옥상에서 수박)이지 촬영 장면이 아니다
#   visual_imagery                    : 곡 분위기에 맞게 **창작한** 이미지다(refine_data.py)
#   cover_url                         : 앨범 표지
#   youtube_url                       : 링크가 있다는 사실. 공식 채널의 자동 생성 음원 업로드일 수 있다
#   search_style_summary / *_tags     : 분위기 요약·태그
# 장면 자료를 붙이게 되면(URL·타임스탬프·관찰 내용, 또는 문서 인용·버전) 그 필드 이름을 여기에 적고,
# `has_mv_scene_evidence`가 곡·판본·주장이 연결되는지까지 보도록 넓힌다. 모델이 말한
# confidence·"verified"는 자료가 아니다 — 모델 출력은 이 함수의 입력이 아니다.
MV_SCENE_EVIDENCE_FIELDS: tuple[str, ...] = ()


def has_mv_scene_evidence(candidates: Iterable[Any]) -> bool:
    """후보 중 한 곡이라도 MV 장면 출처 자료를 갖고 있는가. 현재 계약으로는 항상 False다."""
    if not MV_SCENE_EVIDENCE_FIELDS:
        return False
    for track in candidates:
        for field in MV_SCENE_EVIDENCE_FIELDS:
            if getattr(track, field, None):
                return True
    return False


# ---------------------------------------------------------------------------
# 결정
# ---------------------------------------------------------------------------


def rerank_skip_reason(
    reranker: Any,
    analysis: QueryAnalysis,
    candidates: Sequence[Any] = (),
) -> str:
    """이 요청의 리랭킹을 정책으로 생략해야 하면 그 사유(RERANK_SKIP_*), 아니면 빈 문자열.

    모델을 부르기 **전에** 판정하고, 추가 모델 호출 없이 끝난다. 사유가 있으면 호출부는
    모델을 부르지 않고 지금 후보 순서(보정·거절 제외 반영)를 그대로 내보낸다.

    백엔드별 동의: `skips_image_dominant`, `skips_unverifiable_mv_scene`. 둘 다 Gemini listwise만 켠다.
    """
    if should_skip_rerank_for_image(reranker, analysis):
        return RERANK_SKIP_IMAGE_DOMINANT
    if (
        getattr(reranker, "skips_unverifiable_mv_scene", False)
        and is_specific_mv_scene_query(analysis.original_query)
        and not has_mv_scene_evidence(candidates)
    ):
        return RERANK_SKIP_MV_SCENE_UNVERIFIABLE
    return ""
