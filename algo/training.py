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

import ray

from .model import OpeningsModel, to_utc_ns


def window_open(value, contact_sensor: bool) -> bool:
    if contact_sensor:
        return not bool(value)
    return bool(value)


# An earlier operator version recorded every message with the window open; its points closer than this to
# the one before belong to the same opening.
SAME_OPENING_NS = 5 * 60 * 10**9


def _rising_edges(frame, contact_sensor: bool) -> typing.List[int]:
    """The times at which the window went from closed to open; an open first sample has no known state before it."""
    samples = sorted((to_utc_ns(t), window_open(value, contact_sensor)) for t, value in zip(frame["time"], frame["window_open"]))
    openings = []
    was_open = None
    for t, is_open in samples:
        if is_open and was_open is False:
            openings.append(t)
        was_open = is_open
    return openings


def first_of_runs(openings: typing.Iterable[int]) -> typing.List[int]:
    """One point per opening: a point within SAME_OPENING_NS after the previous one is dropped."""
    result = []
    last = None
    for t in sorted(set(openings)):
        if last is None or t - last > SAME_OPENING_NS:
            result.append(t)
        last = t
    return result


def _read_history(datasets, contact_sensor: bool, notes: typing.List[str]):
    """The window openings in the history and the earliest time it covers, or None without history."""
    if datasets is None:
        return None
    try:
        frames = [(ray.get(ds) if isinstance(ds, ray.ObjectRef) else ds).to_pandas() for ds in datasets]
    except Exception as ex:
        notes.append(f"no history available, training on stored openings only: {type(ex).__name__}: {ex}")
        return None
    frames = [f[["time", "window_open"]].dropna(subset=["time", "window_open"])
              for f in frames if "time" in f.columns and "window_open" in f.columns]
    frames = [f for f in frames if not f.empty]
    if not frames:
        notes.append("history is empty, training on stored openings only")
        return None
    # Per dataset, so the samples of one input do not interleave with those of another.
    openings = [t for f in frames for t in _rising_edges(f, contact_sensor)]
    return openings, min(to_utc_ns(t) for f in frames for t in f["time"])


def fit_openings_model(datasets, others: typing.List[int], now: int, contact_sensor: bool,
                       legacy_imported: bool, retention_ns: int) -> typing.Tuple[OpeningsModel, typing.List[str]]:
    """
    The model from the input history (None if it could not be requested) and the openings known
    outside it, with the log lines for the operator; runs as a Ray task.
    """
    notes = []
    others = first_of_runs(others)
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
