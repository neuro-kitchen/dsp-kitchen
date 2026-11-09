# Documentation site

Starlight (Astro), deployed to <https://neuro-kitchen.github.io/dsp-kitchen/> on pushes to `main`.

```sh
npm install
npm run dev      # http://localhost:4321/dsp-kitchen/
npm run build    # dist/
```

Node 22. A build also writes the Changelog page from `../CHANGELOG.md` and copies the committed
figures into `public/` (`scripts/changelog.mjs`, `scripts/figures.mjs`; both outputs are git-ignored).

## Generated pages: don't edit them

Each carries a comment naming its generator. Run these from the repository root:

| Pages | Generator | Source |
|---|---|---|
| Python API | `uv run --no-project --with griffe==2.4.1 python -I docs/scripts/python_api.py` | Package source and stubs |
| CLI, sorter settings, pipeline spec | `python3 docs/scripts/cli_reference.py` (needs `dspk` and `dspk-cli-sorter` built with `--profile validate`) | The programs' `--help` and `spec schema` |
| Course | `python3 docs/scripts/course.py` | `playground/*/*.py` and their figures |
| Validation: against SciPy | `python3 docs/scripts/benchmarks.py` | `benchmarks/signal-operations/results/` |

CI fails when the Python API, Course or SciPy pages are out of date. The CLI pages need the built
programs, so regenerate them when commands change.

Landing-page strips: `uv run python docs/scripts/strips.py`.

## Structure

One folder per topic in `src/content/docs/`; topics are header tabs (`src/data/nav.mjs`,
`astro.config.mjs`). Keep pages to one idea. A plugin gets `plugins/<name>/` with the shape of Synapse's:
overview, learn, reference, validation.
