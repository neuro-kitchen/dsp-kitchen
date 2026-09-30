"""Read an NWB-Zarr store written by neuro-convert back with the reference Python stack.

Usage:
    uv run --no-project --with pynwb --with hdmf-zarr --with nwbinspector \
        tests/validate_nwb.py <store.nwb.zarr> [--small]

--small also checks the exact values written by tests/outputs_nwb.rs.
"""

import sys

import numpy as np
from hdmf_zarr.nwb import NWBZarrIO


def main(path: str, small: bool) -> None:
    with NWBZarrIO(path, mode="r") as io:
        nwb = io.read()
        print(f"session:     {nwb.session_description!r}")
        print(f"start:       {nwb.session_start_time}")
        print(f"subject:     {nwb.subject.subject_id} {nwb.subject.species} sex={nwb.subject.sex} age={nwb.subject.age}")
        print(f"devices:     {list(nwb.devices)}")
        print(f"groups:      {list(nwb.electrode_groups)}")
        print(f"electrodes:  {len(nwb.electrodes)} rows, columns {list(nwb.electrodes.colnames)}")
        for name, obj in nwb.acquisition.items():
            shape = obj.data.shape
            rate = getattr(obj, "rate", None)
            first = np.asarray(obj.data[:2]).ravel()[:3]
            print(f"acquisition/{name:8} {type(obj).__name__:16} {shape} rate={rate} unit={obj.unit} first={first}")
        for name, t in nwb.intervals.items():
            print(f"intervals/{name:10} {len(t)} rows, columns {list(t.colnames)}")
        for name, t in nwb.analysis.items():
            print(f"analysis/{name:11} {len(t)} rows, columns {list(t.colnames)}")

        if small:
            emg = nwb.acquisition["EMG"]
            assert emg.data.shape == (1000, 3)
            assert np.array_equal(emg.data[5], [5, 1005, 2005]), emg.data[5]
            assert emg.conversion == 1e-6 and emg.rate == 1000.0
            assert list(emg.electrodes.data[:]) == [0, 1, 2]
            assert emg.electrodes.table.name == "electrodes"
            assert nwb.electrodes["group"][0].name == "EMG"
            assert nwb.electrodes["group"][0].device.name == "RZ2(1)"
            temp = nwb.acquisition["Temp"]
            assert temp.data.shape == (1000,) and temp.data[10] == 5.0
            met = nwb.intervals["MET"]
            assert list(met["start_time"][:]) == [0.5, 1.5] and list(met["value"][:]) == [1.0, 2.0]
            tick = nwb.acquisition["Tick"]
            assert tick.rate == 1.0 and tick.starting_time == 0.0 and tick.timestamps is None
            imp = list(nwb.electrodes["imp"][:])
            assert imp[0] == 960.0 and np.isnan(imp[1]) and np.isnan(imp[2]), imp
            assert nwb.acquisition["eS1p"].data.shape == (1, 2)
            assert list(nwb.analysis["Z_EMG"]["note"][:]) == ["ok", "n/a"]
            assert str(nwb.session_start_time) == "2025-02-26 15:25:56-05:00"
            print("exact values: OK")

    # Best-practice checks used by DANDI
    from nwbinspector import inspect_nwbfile_object

    with NWBZarrIO(path, mode="r") as io:
        messages = list(inspect_nwbfile_object(io.read()))
    # nwbinspector's compression checks read `.compressor`, which zarr-python 3 removed for
    # Zarr v3 arrays; they crash on every v3 store (pynwb's own output too), so drop them.
    upstream = [m for m in messages if "not available for Zarr format 3" in m.message]
    messages = [m for m in messages if m not in upstream]
    if upstream:
        print(f"nwbinspector: ignored {len(upstream)} compression-check crashes (upstream Zarr v3 issue)")
    print(f"nwbinspector: {len(messages)} messages")
    for m in messages:
        print(f"  [{m.importance.name}] {m.check_function_name}: {m.message} ({m.location})")


if __name__ == "__main__":
    main(sys.argv[1], "--small" in sys.argv)
