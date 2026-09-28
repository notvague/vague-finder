# Synthetic parser fixtures

`song.html` and `album.html` contain fictitious parser data. They are NOT saved
Namuwiki pages and do not establish the live site's DOM or facts.

`song_meta.json` contains the core metadata shape needed by the isolated
backfill tests. Keeping it here makes tests independent of experiments and the
real data directory. The lyrics, album description and comments are synthetic
placeholders, not copied from the source pages; only factual metadata (IDs,
title, artist, release date) is real, because the matching tests depend on it. Tests create dummy audio and cover files in a temporary
directory when media validation is part of the case.

`test_collection.py` generates synthetic deep DIV wrappers and numbered
sections based on the supplied DOM diagnostics. Its song names mirror the
requested examples; the prose is test data, not a newly verified source.
Live HTML validation remains distinct from these offline tests.
