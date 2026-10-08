# Manually reviewed Namuwiki pages: completion procedure

Run in the project root from PowerShell. This procedure assumes a human has
checked **every** ID currently in `artifacts/context/unresolved.json` and found
no Namuwiki page that can safely be bound to that song. A page that exists but
has no usable trivia is `no_trivia`; an inaccessible page is not `not_found`.

The included CLI refuses to run when the unresolved queue is no longer exactly
481 `needs_review` IDs, when any other unresolved type exists, or when a raw
`meta.json` no longer has the expected `namuwiki_v3/needs_review` state. It does
not crawl or open Qdrant. Its default mode is read-only. `--apply` copies all
original metadata into `data/context/manual_review_backups/` before the first
update and writes a private review manifest. Do not commit the backups, raw
metadata or the ID list to the public repository.

First inspect the status breakdown. The current unresolved queue includes
`document_structure_error` and `multiple_equally_strong_song_sections`; only
include these IDs if the manual check really established no bindable song page.

```powershell
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.resolve_manual_not_found --expected-count 481
```

If the printed counts and the 481 individual decisions match the reviewed
queue, apply the transition:

```powershell
docker compose stop backend
docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.resolve_manual_not_found --expected-count 481 --apply --confirm-all-reviewed-no-page
if ($LASTEXITCODE -ne 0) { throw 'Manual Context resolution failed' }
```

Then rebuild **all 481** missing artifacts. The backfill CLI defaults to a
40-song batch, so `--limit 3016` is intentional. No network calls are made by
`--artifacts-only`.

```powershell
docker compose run --rm --no-deps -T backend python -m src.crawler.scripts_py.backfill_namuwiki_context --artifacts-only --limit 3016
if ($LASTEXITCODE -ne 0) { throw 'Context artifacts failed' }

docker compose run --rm --no-deps -T backend python -m experiments.namuwiki.audit_raw_namuwiki_meta --raw-dir data/raw
if ($LASTEXITCODE -ne 0) { throw 'Raw Context schema audit failed' }

docker compose run --rm --no-deps -T backend python -m src.embedding.cli.embed_context_dense --require-complete-context
if ($LASTEXITCODE -ne 0) { throw 'Context Dense regeneration failed' }

docker compose run --rm --no-deps -T backend python -m src.embedding.cli.fit_context_bm25 --require-complete-context
if ($LASTEXITCODE -ne 0) { throw 'Context BM25 regeneration failed' }

docker compose run --rm --no-deps -T backend python -m src.vector_db.cli.qdrant_load_context --dry-run --require-complete-context
if ($LASTEXITCODE -ne 0) { throw 'Final Context dry-run failed' }
```

Only after the dry run reports `source_scope_complete: true`, publish to the
stopped local Qdrant and restart the backend even on failure:

```powershell
try {
    docker compose run --rm --no-deps -T backend python -m src.vector_db.cli.qdrant_load_context --require-complete-context
    if ($LASTEXITCODE -ne 0) { throw 'Final Context load failed' }
} finally {
    docker compose up -d backend
}
```

Expected values if none of the 478 positive pages changed: 3,016 completed
artifacts; `ok=478`, `no_trivia=692`, `not_found=1846`, `needs_review=0`;
3,995 Dense records and 478 Sparse song profiles. If these differ, inspect the
reason before loading. A successful strict load establishes catalogue coverage,
not the ranking quality of all queries.

For an interrupted update, the private backup folder contains each original
`<song_id>.meta.json` plus `review_manifest.json`. Do not retry against the
stale unresolved list; restore originals or finish reconciling the IDs before
rebuilding artifacts.
