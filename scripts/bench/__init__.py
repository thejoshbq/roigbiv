"""Benchmark measurement tools for the ROIGBIV agent harness.

Scores centroid proposers and deciders against a frozen benchmark that lives
outside this repo (``ROIGBIV_BENCH_ROOT``). Nothing here reads lab data from
the repo, and nothing writes to the benchmark.

Modules:

- ``schema``: frozen dataclasses for the manifest, annotations and scores
- ``manifest``: load and validate a benchmark manifest, load annotations
- ``score``: centroid recall and precision per FOV and per lens, inter-rater mode
- ``split_audit``: dev/test leak check and the G0 size gate
- ``activity_images``: activity-summary and temporal SD images from a movie
- ``union``: candidate-pool recall and unique recall per proposer
- ``decide``: scores-only decider, fit on dev and scored on test
- ``cli``: ``python scripts/bench/cli.py {score,audit,inter-rater}``

Imports resolve with ``scripts/`` on ``sys.path`` (``bench.*``,
``centroid_bakeoff.*``), the same convention the bake-off packages use.
"""
