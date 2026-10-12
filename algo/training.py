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

__all__ = ("fit_openings_model", "train_openings_model", "window_open")

import typing

import pandas as pd
import ray

from .model import OpeningsModel, to_utc_ns


def window_open(value, contact_sensor: bool) -> bool:
    if contact_sensor:
        return not bool(value)
    return bool(value)


def _read_history(datasets, contact_sensor: bool, notes: typing.List[str]):
    """The window openings in the history and the earliest time it covers, or None without history."""
    if datasets is None:
        return None
    try:
        frames = [(ray.get(ds) if isinstance(ds, ray.ObjectRef) else ds).to_pandas() for ds in datasets]
    except Exception as ex:
        notes.append(f"no history available, training on stored openings only: {type(ex).__name__}: {ex}")
        return None
    frames = [f[["time", "window_open"]] for f in frames if "time" in f.columns and "window_open" in f.columns]
    frame = pd.concat(frames, ignore_index=True).dropna(subset=["time", "window_open"]) if frames else None
    if frame is None or frame.empty:
        notes.append("history is empty, training on stored openings only")
        return None
    times = [to_utc_ns(t) for t in frame["time"]]
    openings = [t for t, value in zip(times, frame["window_open"]) if window_open(value, contact_sensor)]
    return openings, min(times)


def fit_openings_model(datasets, others: typing.List[int], now: int, contact_sensor: bool,
                       legacy_imported: bool, retention_ns: int) -> typing.Tuple[OpeningsModel, typing.List[str]]:
    """
    The model from the input history (None if it could not be requested) and the openings known
    outside it, with the log lines for the operator; runs as a Ray task.
    """
    notes = []
    history = _read_history(datasets, contact_sensor, notes)
    if history is None:
        openings = others
    else:
        history_openings, history_start = history
        # An opening carries the device time in the history and the Kafka time everywhere else, so exact
        # deduplication would count it twice; where the history reaches, only the history counts.
        openings = history_openings + [t for t in others if t < history_start or t > now]

    openings = [t for t in set(openings) if now - t <= retention_ns]
    notes.append(f"trained on {len(openings)} window openings")
    return OpeningsModel.from_openings(openings, trained_until=now, legacy_imported=legacy_imported), notes


train_openings_model = ray.remote(fit_openings_model)
