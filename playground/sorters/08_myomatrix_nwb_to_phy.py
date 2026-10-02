#!/usr/bin/env python3
"""
Myomatrix / EMUsort NWB Spike Sorting Runner (Backward-Compatible Forwarding Script).
Directs calls to the canonical `run_emusort_nwb_to_phy` implementation.
"""

import sys
from pathlib import Path
from importlib import import_module

sys.path.insert(0, str(Path(__file__).parent))
emusort_mod = import_module("08_emusort_nwb_to_phy")

run_myomatrix_nwb_to_phy = emusort_mod.run_emusort_nwb_to_phy
run_emusort_nwb_to_phy = emusort_mod.run_emusort_nwb_to_phy

if __name__ == "__main__":
    emusort_mod.main()
