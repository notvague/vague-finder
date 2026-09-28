"""
src/vector_db/cli/admin.py

CLI:
- fetch: id 레코드 확인 (sparse_values 포함 여부 확인)
- stats: namespace count 확인
- delete-ids: id 목록 삭제
- delete-all: namespace 전체 삭제
- delete-filter: metadata filter로 삭제

실행 예시 (docker compose up 상태에서):
  docker compose exec backend python -u -m src.vector_db.cli.admin fetch --kind text --id 36717264 --namespace dev
  docker compose exec backend python -u -m src.vector_db.cli.admin stats --kind text --namespace dev
  docker compose exec backend python -u -m src.vector_db.cli.admin delete-all --kind text --namespace dev
"""
from __future__ import annotations

import argparse
import json
from pprint import pprint

from src.vector_db.admin import (
    fetch_one,
    describe_stats,
    delete_ids,
    delete_all,
    delete_by_filter,
)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # fetch
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--kind", choices=["text", "image", "audio"], required=True)
    p_fetch.add_argument("--id", required=True)
    p_fetch.add_argument("--namespace", default="dev")

    # stats
    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--kind", choices=["text", "image", "audio"], required=True)
    p_stats.add_argument("--namespace", default="dev")

    # delete ids
    p_del_ids = sub.add_parser("delete-ids")
    p_del_ids.add_argument("--kind", choices=["text", "image", "audio"], required=True)
    p_del_ids.add_argument("--ids", nargs="+", required=True)
    p_del_ids.add_argument("--namespace", default="dev")

    # delete all
    p_del_all = sub.add_parser("delete-all")
    p_del_all.add_argument("--kind", choices=["text", "image", "audio"], required=True)
    p_del_all.add_argument("--namespace", default="dev")

    # delete by filter
    p_del_filter = sub.add_parser("delete-filter")
    p_del_filter.add_argument("--kind", choices=["text", "image", "audio"], required=True)
    p_del_filter.add_argument("--filter", required=True, help='예: \'{"artist":{"$eq":"화사 (HWASA)"}}\'')
    p_del_filter.add_argument("--namespace", default="dev")

    args = p.parse_args()
    ns = args.namespace

    if args.cmd == "fetch":
        vec = fetch_one(args.kind, args.id, namespace=ns)
        pprint(vec)
        return

    if args.cmd == "stats":
        st = describe_stats(args.kind, namespace=ns)
        pprint(st)
        return

    if args.cmd == "delete-ids":
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        delete_ids(args.kind, ids, namespace=ns)
        print(f"[OK] deleted ids={len(ids)} kind={args.kind} ns={ns}")
        return

    if args.cmd == "delete-all":
        delete_all(args.kind, namespace=ns)
        print(f"[OK] deleted ALL kind={args.kind} ns={ns}")
        return

    if args.cmd == "delete-filter":
        filter_ = json.loads(args.filter)
        delete_by_filter(args.kind, filter_, namespace=ns)
        print(f"[OK] deleted by filter kind={args.kind} ns={ns}")
        return
    
    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()