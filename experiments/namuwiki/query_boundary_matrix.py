"""Synthetic source-bound queries; no corpus, answers or evaluation labels.

Cross independent Context, heard-sound and album-art evidence. This is a parser
regression matrix, not a replacement for paired OFF/ON retrieval evaluation.
"""

from dataclasses import dataclass
from itertools import permutations, product


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    query: str
    context: bool
    image: str = ""
    audio: str = ""


EVENTS = (
    ("rendition_trend", "어떤 가수의 커버가 화제가 돼 역주행한 곡이야"),
    ("cover_verb", "다른 가수가 커버해서 유명해진 노래야"),
    ("cover_singing", "다른 가수가 커버를 불러 원곡이 다시 알려진 곡이야"),
    ("cover_dance", "방송에서 커버 댄스한 곡이야"),
    ("cover_contest", "커버 대회에서 우승한 곡이야"),
    ("video_scene", "뮤비에 특이한 시계 건물이 나온 노래야"),
    ("lyric_revision", "작사가가 가사 마지막 구절을 바꾼 노래야"),
    ("commercial_adaptation", "광고에서 가사를 바꿔 부른 노래야"),
    ("game_license", "게임에 수록됐다가 판권 만료로 삭제된 곡이야"),
    ("media_use", "드라마 가상작품 OST였던 곡이야"),
)
ART = (
    ("no_art", "", ""),
    ("album_cover", "앨범 커버에는 검은 배경에 꽃 그림이 있었어",
     "Album artwork with drawn flowers on a black background."),
    ("bare_cover", "커버에 흰 글씨와 빨간 꽃이 찍혀 있었어",
     "Cover artwork with white lettering and red flowers."),
    ("implicit_art", "종이 질감에 손글씨 제목과 파란 꽃 그림을 기억해",
     "A paper texture, handwritten title and blue floral illustration."),
)
SOUND = (
    ("no_sound", "", ""),
    ("instruments", "오르골이랑 바이올린 소리가 들리는 발라드였어",
     "A music box melody with violin accompaniment in a slow ballad."),
    ("chorus", "중간에 남자 코러스가 들렸어",
     "Male background vocals enter in the middle."),
    ("absence", "피아노가 작게 들리고 드럼 소리가 없었어",
     "Soft piano accompaniment without drums or percussion."),
)
JOINS = (("sentences", ". "), ("commas", ", "), ("lines", "\n"))


def boundary_cases() -> tuple[BoundaryCase, ...]:
    values = []
    seen = set()
    for (event_name, event), (art_name, art, image), (sound_name, sound, audio), (join_name, join) in product(EVENTS, ART, SOUND, JOINS):
        clauses = tuple((name, part) for name, part in (("context", event), ("image", art), ("audio", sound)) if part)
        for order in permutations(clauses):
            query = join.join(part for _, part in order) + "."
            if query in seen:
                continue
            seen.add(query)
            order_name = "_".join(name for name, _ in order)
            values.append(BoundaryCase(
                f"{event_name}_{art_name}_{sound_name}_{join_name}_{order_name}",
                query, True, image, audio,
            ))
    # Genuine image and sound questions also need a Context-OFF boundary.
    for (art_name, art, image), (sound_name, sound, audio), (join_name, join) in product(ART, SOUND, JOINS):
        if not art and not sound:
            continue
        clauses = tuple((name, part) for name, part in (("image", art), ("audio", sound)) if part)
        for order in permutations(clauses):
            query = join.join(part for _, part in order) + "."
            if query in seen:
                continue
            seen.add(query)
            order_name = "_".join(name for name, _ in order)
            values.append(BoundaryCase(
                f"control_{art_name}_{sound_name}_{join_name}_{order_name}",
                query, False, image, audio,
            ))
    return tuple(values)
