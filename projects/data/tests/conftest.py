"""Test-environment shims for the data project.

`data/waveforms/utils.py`, `data/fetch/fetch.py` and
`data/segments/segments.py` import gwpy at module scope, but gwpy is not a
declared dependency of this project and is absent from the venv, so every
test module that reaches them fails at *collection* rather than at run time.

conftest is imported before collection, so this is where the shim has to live;
a stub inside a test module is too late for the modules pytest collects
earlier. The import is attempted first so a real gwpy is never shadowed, and
the handler is narrow: it stubs only when gwpy itself is missing. A gwpy that
is installed but broken -- an unimportable dependency of its own, say -- must
surface as the error it is rather than be papered over with stubs.
"""

import sys
from types import ModuleType

try:  # pragma: no cover - depends on the environment, not the code
    import gwpy.timeseries  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "gwpy":
        raise
    _gwpy = ModuleType("gwpy")

    _timeseries = ModuleType("gwpy.timeseries")
    _timeseries.TimeSeries = object
    _timeseries.TimeSeriesDict = object

    _segments = ModuleType("gwpy.segments")
    _segments.DataQualityDict = object
    _segments.DataQualityFlag = object
    _segments.SegmentList = object

    _gwpy.timeseries = _timeseries
    _gwpy.segments = _segments

    sys.modules["gwpy"] = _gwpy
    sys.modules["gwpy.timeseries"] = _timeseries
    sys.modules["gwpy.segments"] = _segments
