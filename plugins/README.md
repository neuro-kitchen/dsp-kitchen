# Plugins

Separate programs the front ends run; they are never linked into them, so each keeps its own
licence and release.

- `cli/`: programs the command line (`dspk`) runs. `dspk-cli-sorter`: every spike sorter in one
  program (GPL-3.0; contains Kilosort4 and EMUsort), run by `dspk sort` and installed by
  `dspk sorters install`. The app runs the same program for its Sort action.
- `app/`: plugins of the desktop app (`dspk-app`); none yet.
