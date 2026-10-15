# Project page

Buildless HTML, CSS, and JavaScript for the Vidur speculative-decoding presentation.
The page reads only the bundled recorded simulator sweep; it does not run Vidur
in a browser or claim measured GPU speedups.

From the repository root:

```bash
python scripts/export_project_results.py         # after changing the sweep CSV
python scripts/export_project_results.py --check # verify the tracked JSON export
python -m http.server 8765 --directory docs/project
```

Open http://localhost:8765. The `Publish project page` workflow deploys this
directory to GitHub Pages on changes to the page or its recorded data.
The `Speculative decoding tests` workflow runs the unit and fake-server suites.
The cache-dependent end-to-end test is skipped unless a trained predictor cache
is explicitly supplied.

Source data: `docs/assets/spec_decode_sweep/results.csv`. All 39 rows are
exported without changing their values. Three loads × (one baseline + three
proposal counts × two acceptance rates × two attention assumptions).
