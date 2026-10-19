"""
   Copyright 2022 InfAI (CC SES)

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

__all__ = ("Operator", )

import typing

import dotenv
dotenv.load_dotenv()

from operator_lib.util import MLOperator, Config, Inputs, Outputs, logger
from operator_lib.util import clock
from operator_lib.util.helpers import provide_historic_data
from operator_lib.util.persistence import load
import os
import json
import datetime
import threading
import collections
import time
import pandas as pd
import ray
from algo import compute_clusters_boundaries, compute_confidence_from_spreading, compute_confidence_by_daily_apperance
from algo.model import OpeningsModel, to_utc_ns, to_wall_time, wall_time_to_utc_ns
from algo.training import train_openings_model, window_open as is_window_open


STATE_FILE = "openings_state.json"
# Written by the operator before it trained models; read once so existing pipelines keep what they learned.
LEGACY_WINDOW_OPENING_TIMES_FILE = "window_opening_times.pickle"
LEGACY_LAST_TIMESTAMP_FILE = "last_timestamp.pickle"

RETENTION_NS = 60 * 24 * 3600 * 10**9  # 60 days
SAVE_INTERVAL_S = 60
# A requested history that cannot be read, as when the Ray cluster has no CPU free, is tried again this
# often, this far apart, before the day counts as trained on the stored openings alone.
HISTORY_RETRY_S = 600
HISTORY_RETRIES = 6


class CustomConfig(Config):
    # Redeclared so the manifest lists it, with the library's default.
    logger_level: str = "warning"

    data_path: str = "/opt/data"

    high_confidence_boundary: int = 600 # in seconds, corresponds to 10 minutes
    low_confidence_boundary: int = 3600 # in seconds, corresponds to 1 hour

    inertia_buffer: int = 10 # in minutes

    confidence_days: int = 7 # in days

    contact_sensor: bool = True

    history_days: float = 60 # in days, read at each training

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)

        # Parsed as before the port; only history_days is new.
        self.high_confidence_boundary = int(self.high_confidence_boundary)
        self.low_confidence_boundary = int(self.low_confidence_boundary)
        self.inertia_buffer = int(self.inertia_buffer)
        self.confidence_days = int(self.confidence_days)
        self.history_days = 60 if str(self.history_days).strip() == "" else float(self.history_days)
        # Read as before the port: any non-empty string, "False" included, switches the inversion on.
        self.contact_sensor = bool(self.contact_sensor)


def _unwrap(model) -> typing.Optional[OpeningsModel]:
    if model is None:
        return None
    if hasattr(model, "unwrap_python_model"):
        model = model.unwrap_python_model()
    if not isinstance(model, OpeningsModel):
        logger.warning(f"ignoring a model of unexpected type {type(model).__name__}")
        return None
    return model


class OperatorInputs(Inputs):
    window_open: bool


class OperatorOutputs(Outputs):
    stopping_time: str
    overall_confidence: str
    timestamp: str


class Operator(MLOperator):
    configType = CustomConfig
    inputs = OperatorInputs
    outputs = OperatorOutputs

    def __init__(self):
        # stop() runs on the watchdog thread while infer() may be running.
        self._lock = threading.Lock()
        self._loaded = False
        self.recent = []  # openings seen live and not yet in a trained model, UTC ns
        self.window_was_open = None  # state of the last in-order message, None while unknown
        self.last_timestamp = None  # time of the last message, UTC ns
        self.retrain = False
        self.trained_day = None  # Berlin date of the wall clock at the last training
        self._retry_at = None  # monotonic time of the next training after an unreadable history
        self._history_retries = 0
        self._saved_at = None  # monotonic time of the last state save

    def stop(self):
        super().stop()
        with self._lock:
            if self._loaded:
                self._save_state()

    def _ensure_loaded(self):
        # Lazy because MLOperator.init() calls train() before it returns.
        with self._lock:
            if self._loaded:
                return
            try:
                os.makedirs(self.config.data_path, exist_ok=True)
            except OSError as ex:
                logger.error(f"cannot create {self.config.data_path}: {ex}")
            self.recent, self.last_timestamp, self.window_was_open = self._load_state()
            self._loaded = True

    def _load_state(self):
        path = os.path.join(self.config.data_path, STATE_FILE)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                recent = [int(t) for t in state["recent"]]
                last = state["last_timestamp"]
                # Absent in the state of earlier versions.
                was_open = state.get("window_open")
                return (recent, (None if last is None else int(last)),
                        was_open if isinstance(was_open, bool) else None)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as ex:
                logger.error(f"ignoring unreadable state in {path}: {ex}")
                return [], None, None
        try:
            legacy_last = load(self.config.data_path, LEGACY_LAST_TIMESTAMP_FILE)
        except Exception as ex:
            logger.error(f"ignoring unreadable {LEGACY_LAST_TIMESTAMP_FILE}: {ex}")
            legacy_last = None
        return [], (None if legacy_last is None else wall_time_to_utc_ns(legacy_last)), None

    def _save_state(self):
        # Caller holds the lock. Replacing the file keeps the previous state if writing fails halfway.
        path = os.path.join(self.config.data_path, STATE_FILE)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump({"recent": self.recent, "last_timestamp": self.last_timestamp,
                           "window_open": self.window_was_open}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            self._saved_at = time.monotonic()
        except OSError as ex:
            logger.error(f"cannot save state to {path}: {ex}")

    def _read_legacy_openings(self) -> typing.Optional[typing.List[int]]:
        """The openings stored before the operator trained models; None if the file is unreadable."""
        try:
            stored = load(self.config.data_path, LEGACY_WINDOW_OPENING_TIMES_FILE)
            if stored is None:
                return []
            openings = [wall_time_to_utc_ns(ts) for key in ("weekday", "weekend") for ts in stored.get(key, [])]
        except Exception as ex:
            logger.error(f"cannot read {LEGACY_WINDOW_OPENING_TIMES_FILE}, retrying at the next training: {ex}")
            return None
        logger.info(f"imported {len(openings)} window openings from {LEGACY_WINDOW_OPENING_TIMES_FILE}")
        return openings

    def _request_history(self):
        """References to the input history, read by the training task; None if it cannot be requested."""
        try:
            return provide_historic_data(datetime.timedelta(days=self.config.history_days))
        except Exception as ex:
            logger.info(f"no history available, training on stored openings only: {type(ex).__name__}: {ex}")
            return None

    def _window_open(self, value) -> bool:
        return is_window_open(value, self.config.contact_sensor)

    def train(self, model, train_logger):
        self._ensure_loaded()
        previous = _unwrap(model)
        now = to_utc_ns(clock.now())
        datasets = self._request_history()

        with self._lock:
            consumed = list(self.recent)
        others = list(consumed)
        if previous is not None:
            others += previous.weekday + previous.weekend

        legacy_imported = previous is not None and previous.legacy_imported
        if not legacy_imported:
            legacy = self._read_legacy_openings()
            if legacy is not None:
                others += legacy
                legacy_imported = True

        openings_model, notes, history_failed = ray.get(train_openings_model.remote(
            datasets, others, now, self.config.contact_sensor, legacy_imported, RETENTION_NS))
        for note in notes:
            logger.info(note)
        # Read by the content test to tell a training on real history from one on nothing.
        if train_logger is not None:
            train_logger.log_metrics({"window_openings": len(openings_model.weekday) + len(openings_model.weekend)})
        # The live openings are in the model now (or replaced by the history); infer must not count them twice.
        with self._lock:
            left = collections.Counter(consumed)
            remaining = []
            for t in self.recent:
                if left[t] > 0:
                    left[t] -= 1
                else:
                    remaining.append(t)
            self.recent = remaining
            if history_failed and self._history_retries < HISTORY_RETRIES:
                # The day stays untrained, so the next training reads the history again.
                self._history_retries += 1
                self._retry_at = time.monotonic() + HISTORY_RETRY_S
                logger.warning(f"history unreadable, training again in {HISTORY_RETRY_S // 60} minutes "
                               f"(retry {self._history_retries} of {HISTORY_RETRIES})")
            else:
                self._history_retries = 0
                self._retry_at = None
                self.trained_day = to_wall_time(now).date()
            self._save_state()
        return openings_model

    def need_retraining(self, model) -> bool:
        with self._lock:
            retrain = self.retrain or (self._retry_at is not None and time.monotonic() >= self._retry_at)
            self.retrain = False
            if retrain:
                self._retry_at = None
        return retrain

    def infer(self, model, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        self._ensure_loaded()
        utc_ns = to_utc_ns(timestamp)
        # German wall time without zone, as the clustering expects.
        current_timestamp = to_wall_time(utc_ns)
        window_open = self._window_open(data["window_open"])
        logger.debug(f"{current_timestamp}:  Window open: {window_open}!")

        openings_model = _unwrap(model)
        with self._lock:
            # A late message must not move the day back, or the next message would start the same day again;
            # nor does it say whether the window opened now.
            in_order = self.last_timestamp is None or utc_ns > self.last_timestamp
            changed = in_order and window_open != self.window_was_open
            # The input repeats the state with every message; only the change from closed to open is an opening.
            if changed and window_open and self.window_was_open is False:
                self.recent.append(utc_ns)
            if in_order:
                self.window_was_open = window_open
            new_day = self.last_timestamp is not None and self.check_for_new_day(to_wall_time(self.last_timestamp), current_timestamp)
            if in_order:
                self.last_timestamp = utc_ns
            if new_day:
                # At most one training per day of the wall clock, so replaying a backlog does not train per replayed day.
                self.retrain = self.trained_day != to_wall_time(to_utc_ns(clock.now())).date()
                self.recent = [t for t in self.recent if utc_ns - t <= RETENTION_NS]
            if new_day or changed or (window_open and (self._saved_at is None
                                            or time.monotonic() - self._saved_at >= SAVE_INTERVAL_S)):
                self._save_state()
            recent = list(self.recent)

        if not new_day:
            return None, None, None

        weekend = self.check_if_weekend(current_timestamp)
        considered = {t for t in recent if self.check_if_weekend(to_wall_time(t)) == weekend}
        if openings_model is not None:
            considered.update(openings_model.openings(weekend))
        considered_timestamps = [to_wall_time(t) for t in sorted(considered) if utc_ns - t <= RETENTION_NS]

        if len(considered_timestamps) <= 2:
            logger.debug({"stopping_time": "Not enough data!",
                          "overall_confidence": str(0),
                          "timestamp": self.prepare_output_timestamp(current_timestamp)})
            return None, None, None

        inertia_buffer = pd.Timedelta(self.config.inertia_buffer, "min")
        clusters_boundaries, indices = compute_clusters_boundaries(considered_timestamps)
        current_day = current_timestamp.floor("d")
        confidence_list = []
        for c in clusters_boundaries.keys():
            pair_of_boundaries = clusters_boundaries[c]
            ts_in_cluster = [considered_timestamps[i] for i in indices[c]]
            confidence_by_spreading = compute_confidence_from_spreading(ts_in_cluster, self.config.high_confidence_boundary, self.config.low_confidence_boundary)
            confidence_by_daily_appearance = compute_confidence_by_daily_apperance(current_timestamp, considered_timestamps, pair_of_boundaries, confidence_days=self.config.confidence_days)
            overall_confidence = confidence_by_spreading * confidence_by_daily_appearance

            confidence_list.append({"stopping_time": self.prepare_output_timestamp(pd.Timestamp.combine(current_day, pair_of_boundaries[0]) - inertia_buffer),
                                    "overall_confidence": str(overall_confidence),
                                    "timestamp": self.prepare_output_timestamp(current_timestamp)})
        logger.debug(f"Results for next day: {confidence_list}")
        # No output time of its own, as before the port: the producer stamps it.
        return None, confidence_list, None

    def check_for_new_day(self, last_timestamp: pd.Timestamp, current_timestamp: pd.Timestamp):
        if current_timestamp.date() > last_timestamp.date():
            return True
        else:
            return False

    def check_if_weekend(self, current_timestamp: pd.Timestamp):
        if current_timestamp.weekday() <= 4: # Mo, Tu, Wd, Th, Fr
            return False
        else:
            return True

    def prepare_output_timestamp(self, current_timestamp: pd.Timestamp):
        # Wall times computed in Berlin can fall into the DST gap or the repeated hour.
        current_timestamp = current_timestamp.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous=False).tz_convert("Zulu")
        current_timestamp_string = current_timestamp.isoformat().replace('+00:00', 'Z')
        return current_timestamp_string

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="automatic-heating-operator", git_info_file='git_commit')
