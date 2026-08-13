"""Why did two sessions of the same FOV fail to match?

``registry_match.json`` reports a decision and a posterior. When the decision
is wrong, neither number says which stage lost the correspondence — alignment,
one of the three similarity channels, the pruning cutoff, or the Hungarian
assignment. This package opens the matcher up and measures each stage against
a ground truth derived independently of ROICaT.

Ground truth (``ground_truth.py``) comes from the centroids themselves: a
validated rigid shift between mean projections plus Hungarian assignment. That
is deliberately a *different* method from the one under test, so agreement
between them means something.

Read-only. Touches no registry database and writes nothing outside the report
path it is given.

Usage: ``python -m scripts.registry_diagnose.run_diagnose --help``
"""
