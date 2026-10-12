"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("OpeningsModel", "TIMEZONE", "to_utc_ns", "to_wall_time", "wall_time_to_utc_ns", "is_weekend")

import typing

import mlflow.pyfunc
import pandas as pd

from .create_clustering import compute_clusters_boundaries

TIMEZONE = "Europe/Berlin"


def to_utc_ns(value) -> int:
    """Nanoseconds since the epoch of a time; a value without zone counts as UTC."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value)


def to_wall_time(utc_ns: int) -> pd.Timestamp:
    """Berlin wall time without zone, the form the clustering works on."""
    return pd.Timestamp(int(utc_ns), tz="UTC").tz_convert(TIMEZONE).tz_localize(None)


def wall_time_to_utc_ns(wall_time) -> int:
    ts = pd.Timestamp(wall_time)
    if ts.tzinfo is not None:
        return int(ts.value)
    # Either reading of the repeated autumn hour maps back to the same wall time, which is all the clustering sees.
    return int(ts.tz_localize(TIMEZONE, ambiguous=False, nonexistent="shift_forward").value)


def is_weekend(wall_time: pd.Timestamp) -> bool:
    return wall_time.weekday() > 4


class OpeningsModel(mlflow.pyfunc.PythonModel):
    """
    The window openings of the last 60 days, split by day type. Times are UTC
    nanoseconds as plain ints, so the pickle does not depend on the pandas version.
    """

    def __init__(self, weekday: typing.Iterable[int] = (), weekend: typing.Iterable[int] = (),
                 trained_until: typing.Optional[int] = None, legacy_imported: bool = False):
        self.weekday = sorted({int(t) for t in weekday})
        self.weekend = sorted({int(t) for t in weekend})
        # Wall clock of the training that built the model.
        self.trained_until = None if trained_until is None else int(trained_until)
        self.legacy_imported = bool(legacy_imported)

    @classmethod
    def from_openings(cls, openings: typing.Iterable[int], trained_until: typing.Optional[int] = None,
                      legacy_imported: bool = False) -> "OpeningsModel":
        weekday, weekend = [], []
        for t in openings:
            (weekend if is_weekend(to_wall_time(t)) else weekday).append(t)
        return cls(weekday, weekend, trained_until, legacy_imported)

    def openings(self, weekend: bool) -> typing.List[int]:
        return self.weekend if weekend else self.weekday

    def predict(self, context, model_input, params=None):
        """
        For each given time, the cluster boundaries (Berlin wall times) of the
        openings on its day type, or an empty list with two openings or fewer.
        """
        times = model_input
        if isinstance(times, pd.DataFrame):
            times = times.iloc[:, 0]
        if not isinstance(times, (list, tuple, pd.Series, pd.Index)):
            times = [times]
        result = []
        for t in times:
            wall_times = [to_wall_time(o) for o in self.openings(is_weekend(to_wall_time(to_utc_ns(t))))]
            if len(wall_times) <= 2:
                result.append([])
                continue
            boundaries, _ = compute_clusters_boundaries(wall_times)
            result.append([{"start": b[0].isoformat(), "end": b[1].isoformat()} for b in sorted(boundaries.values())])
        return result
