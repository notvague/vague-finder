"""저장된 질의 분석을 주입해 evaluate_search_accuracy를 돌린다."""
import asyncio, json, sys
import src.retrieval.evaluate_search_accuracy as ev
from src.backend.schemas.query import QueryAnalysis

cache_path, *rest = sys.argv[1:]
cache = json.load(open(cache_path, encoding="utf-8"))

class CachedAnalyzer:
    def analyze(self, query):
        if query not in cache:
            raise KeyError(f"분석 캐시에 없는 질의: {query[:40]}")
        return QueryAnalysis.model_validate(cache[query])   # 매번 새 객체

ev.get_query_analyzer = lambda: CachedAnalyzer()
asyncio.run(ev.evaluate(ev.build_parser().parse_args(rest)))
