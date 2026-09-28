"""Search quality evaluation package.

- schema.py    : EvalQuery / EvalSet Pydantic 정의 (정답·오답·메타)
- loader.py    : queries.json 로드 + 검증 + catalog 교차참조
- build_catalog.py : data/all_songs(test).jsonl → docs/eval/song_catalog.json 재생성
"""
from src.eval.schema import EvalQuery, EvalSet, QueryCategory  
from src.eval.loader import load_eval_set, load_song_catalog  
