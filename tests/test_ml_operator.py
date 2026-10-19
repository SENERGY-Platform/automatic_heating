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

import collections
import contextlib
import datetime
import json
import os
import pickle
import tempfile
import unittest
from unittest import mock

import mlflow
import pandas as pd

from operator_lib.util import clock

import main
from main import Operator, CustomConfig, STATE_FILE, RETENTION_NS, HISTORY_RETRY_S, HISTORY_RETRIES
from algo.model import OpeningsModel, to_utc_ns, wall_time_to_utc_ns
from algo.training import fit_openings_model

BERLIN = "Europe/Berlin"


def berlin(wall_time: str) -> pd.Timestamp:
    """An aware UTC time from a Berlin wall time."""
    return pd.Timestamp(wall_time).tz_localize(BERLIN).tz_convert("UTC")


def ns(wall_time: str) -> int:
    return to_utc_ns(berlin(wall_time))


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame.copy()


class UnreadableDataset:
    """A history whose read fails, as when Ray finds no CPU for it."""

    def to_pandas(self):
        raise RuntimeError("Failed to get allocated resources for data-dataset_407_0 after 10 consecutive failures.")


class FakePyFuncModel:
    """What mlflow.pyfunc.load_model returns: the python model behind unwrap_python_model()."""

    def __init__(self, python_model):
        self.python_model = python_model

    def unwrap_python_model(self):
        return self.python_model


def history_frame(openings, closings=()):
    """Rows as the lib's readers return them: naive UTC time, the mapped value; closed before the first row,
    since an opening needs a known closed state before it."""
    rows = [(berlin(w).tz_localize(None), True) for w in openings]
    rows += [(berlin(w).tz_localize(None), False) for w in closings]
    if rows:
        rows.append((min(t for t, _ in rows) - pd.Timedelta(1, "h"), False))
    return pd.DataFrame(rows, columns=["time", "window_open"]).sort_values("time").reset_index(drop=True)


def weekday_and_weekend_openings(first_day: str, last_day: str):
    openings, closings = [], []
    for day in pd.date_range(first_day, last_day, freq="D"):
        if day.weekday() <= 4:
            minute = day.day % 3
            openings.append(f"{day.date()} 07:0{minute}:00")
            closings.append(f"{day.date()} 07:3{minute}:00")
        else:
            openings.append(f"{day.date()} 10:00:00")
            closings.append(f"{day.date()} 10:30:00")
    return openings, closings


@contextlib.contextmanager
def training_without_ray(history=None, history_error=None):
    """The history request answers with the given datasets and the training task runs in-process."""
    if history_error is not None:
        request = mock.patch("main.provide_historic_data", side_effect=history_error)
    else:
        request = mock.patch("main.provide_historic_data", return_value=history)
    with request, \
         mock.patch("main.train_openings_model", mock.Mock(remote=fit_openings_model)), \
         mock.patch("main.ray.get", side_effect=lambda result: result):
        yield


class OperatorTestCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(clock.set_fixed, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_path = tmp.name

    def operator(self, **config):
        op = Operator()
        op.config = CustomConfig({"data_path": self.data_path, **config})
        return op

    def train(self, op, model=None, history=None, history_error=None, now="2026-01-11 23:00:00"):
        clock.set_fixed(berlin(now).to_pydatetime())
        with training_without_ray(history, history_error):
            return op.train(model, None)

    def open_at(self, op, model, wall_time):
        """A closed message a minute before wall_time, then an open one: the input repeats the state with every
        message, and only the change is an opening."""
        closed = op.config.contact_sensor  # the raw value of a closed window
        before = str(pd.Timestamp(wall_time) - pd.Timedelta(1, "min"))
        self.infer(op, model, before, window_open=closed)
        return self.infer(op, model, wall_time, window_open=not closed)

    def infer(self, op, model, wall_time, window_open=False):
        dt, result, new_model = op.infer(model, {"window_open": window_open}, "s", "d", berlin(wall_time).to_pydatetime())
        self.assertIsNone(new_model)
        return dt, result


class TestTrainAndInfer(OperatorTestCase):
    def setUp(self):
        super().setUp()
        openings, closings = weekday_and_weekend_openings("2025-12-15", "2026-01-11")
        self.history = [FakeDataset(history_frame(openings, closings))]

    def test_monday_gets_one_weekday_cluster_with_inertia_applied(self):
        op = self.operator(contact_sensor="")
        model = FakePyFuncModel(self.train(op, history=self.history))

        self.assertEqual(self.infer(op, model, "2026-01-11 23:30:00"), (None, None))
        dt, result = self.infer(op, model, "2026-01-12 00:05:00")

        # No output time of its own, as before the port.
        self.assertIsNone(dt)
        self.assertEqual(result, [{
            "stopping_time": "2026-01-12T05:50:00Z",
            "overall_confidence": "1.0",
            "timestamp": "2026-01-11T23:05:00Z",
        }])

    def test_saturday_gets_the_weekend_cluster(self):
        op = self.operator(contact_sensor="")
        model = FakePyFuncModel(self.train(op, history=self.history))

        self.infer(op, model, "2026-01-16 23:59:00")
        dt, result = self.infer(op, model, "2026-01-17 00:01:00")

        self.assertEqual(result, [{
            "stopping_time": "2026-01-17T08:50:00Z",
            "overall_confidence": "1.0",
            "timestamp": "2026-01-16T23:01:00Z",
        }])

    def test_contact_sensor_inverts_the_history(self):
        # With the default contact_sensor, False means open: the closings become the openings.
        op = self.operator()
        model = self.train(op, history=self.history)
        points = model.weekday + model.weekend
        self.assertTrue(points)
        self.assertTrue(all(pd.Timestamp(t, tz="UTC").tz_convert(BERLIN).minute >= 30 for t in points))

    def test_wall_time_is_stable_across_the_spring_dst_change(self):
        openings, closings = weekday_and_weekend_openings("2026-03-09", "2026-04-03")
        openings = [o.replace("07:01", "07:00").replace("07:02", "07:00") for o in openings]
        op = self.operator(contact_sensor="")
        model = FakePyFuncModel(self.train(op, history=[FakeDataset(history_frame(openings, closings))],
                                           now="2026-04-05 23:00:00"))

        self.infer(op, model, "2026-04-05 23:30:00")
        _, result = self.infer(op, model, "2026-04-06 00:05:00")

        self.assertEqual([r["stopping_time"] for r in result], ["2026-04-06T04:50:00Z"])

    def test_training_logs_the_number_of_openings(self):
        op = self.operator()
        train_logger = mock.Mock()
        clock.set_fixed(berlin("2026-01-11 23:00:00").to_pydatetime())
        with training_without_ray(self.history):
            model = op.train(None, train_logger)
        openings = len(model.weekday) + len(model.weekend)
        self.assertGreater(openings, 0)
        train_logger.log_metrics.assert_called_once_with({"window_openings": openings})


class TestNotEnoughData(OperatorTestCase):
    def test_two_openings_give_no_output_and_three_do(self):
        op = self.operator()
        self.open_at(op, None, "2026-01-12 07:00:00")
        self.open_at(op, None, "2026-01-13 07:00:00")
        self.assertEqual(self.infer(op, None, "2026-01-14 00:01:00", window_open=True), (None, None))

        self.open_at(op, None, "2026-01-14 07:01:00")
        _, result = self.infer(op, None, "2026-01-15 00:01:00", window_open=True)
        self.assertEqual([r["stopping_time"] for r in result], ["2026-01-15T05:50:00Z"])


class TestColdStart(OperatorTestCase):
    def assert_cold_start_then_live_output(self, **history):
        op = self.operator()
        model = self.train(op, **history)
        self.assertIsInstance(model, OpeningsModel)
        self.assertEqual((model.weekday, model.weekend), ([], []))

        wrapped = FakePyFuncModel(model)
        for day in ("12", "13", "14"):
            self.open_at(op, wrapped, f"2026-01-{day} 07:00:00")
        _, result = self.infer(op, wrapped, "2026-01-15 00:01:00", window_open=True)
        self.assertEqual([r["stopping_time"] for r in result], ["2026-01-15T05:50:00Z"])

    def test_failing_history_read(self):
        self.assert_cold_start_then_live_output(history_error=RuntimeError("no deployment configuration"))

    def test_empty_history(self):
        self.assert_cold_start_then_live_output(history=[FakeDataset(pd.DataFrame(columns=["time", "window_open"]))])

    def test_no_input_topics(self):
        self.assert_cold_start_then_live_output(history=[])

    def test_history_without_the_mapped_column(self):
        self.assert_cold_start_then_live_output(history=[FakeDataset(pd.DataFrame({"time": [pd.Timestamp("2026-01-05")]}))])


class TestOpenings(unittest.TestCase):
    """An opening is the change from closed to open, as the input repeats the state with every message."""

    now = ns("2026-01-11 23:00:00")

    def fit(self, datasets, others=()):
        model, _, _ = fit_openings_model(datasets, list(others), self.now, False, True, RETENTION_NS)
        return model.weekday + model.weekend

    def test_a_state_reported_every_minute_is_one_opening(self):
        times = pd.date_range("2026-01-05 06:50:00", "2026-01-05 08:00:00", freq="min")
        values = [not (pd.Timestamp("2026-01-05 07:00:00") <= t < pd.Timestamp("2026-01-05 07:30:00")) for t in times]
        frame = pd.DataFrame({"time": [berlin(str(t)).tz_localize(None) for t in times], "window_open": values})
        # contact sensor: False is open
        model, _, _ = fit_openings_model([FakeDataset(frame)], [], self.now, True, True, RETENTION_NS)
        self.assertEqual(model.weekday + model.weekend, [ns("2026-01-05 07:00:00")])

    def test_an_open_first_sample_has_no_known_change(self):
        frame = pd.DataFrame({"time": [berlin("2026-01-05 07:00:00").tz_localize(None)], "window_open": [True]})
        self.assertEqual(self.fit([FakeDataset(frame)]), [])

    def test_each_dataset_has_its_own_state(self):
        first = history_frame(["2026-01-05 07:00:00"], ["2026-01-05 07:30:00"])
        second = history_frame(["2026-01-05 07:10:00"], ["2026-01-05 07:40:00"])
        self.assertEqual(self.fit([FakeDataset(first), FakeDataset(second)]),
                         [ns("2026-01-05 07:00:00"), ns("2026-01-05 07:10:00")])

    def test_points_of_the_earlier_version_count_once_per_opening(self):
        per_minute = [ns("2026-01-05 07:00:00") + i * 60 * 10**9 for i in range(30)]
        later = ns("2026-01-05 18:00:00")
        self.assertEqual(self.fit(None, per_minute + [later]), [ns("2026-01-05 07:00:00"), later])


class TestConfig(unittest.TestCase):
    def test_contact_sensor_strings(self):
        # As in v1.0.0: the flow engine hands over strings, only the empty one switches the inversion off.
        for value, expected in (("True", True), ("False", True), ("0", True), ("", False), (False, False)):
            with self.subTest(value=value):
                self.assertIs(CustomConfig({"contact_sensor": value}).contact_sensor, expected)
        self.assertIs(CustomConfig({}).contact_sensor, True)

    def test_old_keys_are_ignored_and_strings_parsed(self):
        config = CustomConfig({
            "init_phase_length": "28",
            "init_phase_level": "d",
            "high_confidence_boundary": "300",
            "low_confidence_boundary": "1800",
            "inertia_buffer": "15",
            "confidence_days": "5",
            "history_days": "30",
        })
        self.assertEqual((config.high_confidence_boundary, config.low_confidence_boundary, config.inertia_buffer,
                          config.confidence_days, config.history_days), (300, 1800, 15, 5, 30.0))
        self.assertFalse(hasattr(config, "init_phase_length"))

    def test_defaults(self):
        config = CustomConfig({})
        self.assertEqual((config.high_confidence_boundary, config.low_confidence_boundary, config.inertia_buffer,
                          config.confidence_days, config.history_days, config.data_path),
                         (600, 3600, 10, 7, 60, "/opt/data"))

    def test_unparsable_numbers_raise_as_before_the_port(self):
        for values in ({"inertia_buffer": "ten"}, {"inertia_buffer": ""}, {"confidence_days": "x"},
                       {"history_days": "x"}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CustomConfig(values)


class TestDayChange(OperatorTestCase):
    def test_retrain_is_flagged_once_per_day_change(self):
        op = self.operator()
        self.infer(op, None, "2026-01-12 10:00:00", window_open=True)
        self.infer(op, None, "2026-01-12 11:00:00", window_open=True)
        self.assertFalse(op.need_retraining(None))

        self.infer(op, None, "2026-01-13 00:01:00", window_open=True)
        self.assertTrue(op.need_retraining(None))
        self.assertFalse(op.need_retraining(None))

        self.infer(op, None, "2026-01-13 00:02:00", window_open=True)
        self.assertFalse(op.need_retraining(None))

    def test_a_late_message_does_not_start_the_day_again(self):
        op = self.operator()
        for day in ("08", "09", "12"):
            self.open_at(op, None, f"2026-01-{day} 07:00:00")
        self.infer(op, None, "2026-01-12 08:00:00", window_open=False)
        _, result = self.infer(op, None, "2026-01-13 00:01:00", window_open=True)
        self.assertIsNotNone(result)

        self.assertEqual(self.infer(op, None, "2026-01-12 23:59:00", window_open=True), (None, None))
        self.assertEqual(self.infer(op, None, "2026-01-13 00:02:00", window_open=True), (None, None))

    def test_replayed_day_changes_train_once_per_day_of_the_wall_clock(self):
        op = self.operator()
        self.train(op, history=[], now="2026-02-01 12:00:00")
        for day in range(12, 20):
            self.infer(op, None, f"2026-01-{day} 07:00:00", window_open=True)
            self.assertFalse(op.need_retraining(None), day)

        clock.set_fixed(berlin("2026-02-02 00:01:00").to_pydatetime())
        self.infer(op, None, "2026-01-20 07:00:00", window_open=True)
        self.assertTrue(op.need_retraining(None))


class TestHistoryRetry(OperatorTestCase):
    def train_at(self, op, monotonic, **history):
        with mock.patch("main.time.monotonic", return_value=monotonic):
            return self.train(op, **history)

    def needs_retraining_at(self, op, monotonic):
        with mock.patch("main.time.monotonic", return_value=monotonic):
            return op.need_retraining(None)

    def test_an_unreadable_history_is_read_again_after_a_pause(self):
        op = self.operator()
        self.train_at(op, 1000.0, history=[UnreadableDataset()])

        self.assertIsNone(op.trained_day)
        self.assertFalse(self.needs_retraining_at(op, 1000.0 + HISTORY_RETRY_S - 1))
        self.assertTrue(self.needs_retraining_at(op, 1000.0 + HISTORY_RETRY_S))
        self.assertFalse(self.needs_retraining_at(op, 1000.0 + 2 * HISTORY_RETRY_S))

    def test_a_read_history_trains_the_day(self):
        op = self.operator()
        self.train_at(op, 1000.0, history=[UnreadableDataset()])
        openings, closings = weekday_and_weekend_openings("2025-12-15", "2026-01-11")
        model = self.train_at(op, 1000.0 + HISTORY_RETRY_S, history=[FakeDataset(history_frame(openings, closings))])

        self.assertTrue(model.weekday)
        self.assertIsNotNone(op.trained_day)
        self.assertFalse(self.needs_retraining_at(op, 1000.0 + 10 * HISTORY_RETRY_S))

    def test_the_day_counts_as_trained_after_the_last_retry(self):
        op = self.operator()
        for attempt in range(HISTORY_RETRIES):
            self.train_at(op, 1000.0 * attempt, history=[UnreadableDataset()])
            self.assertIsNone(op.trained_day, attempt)
        self.train_at(op, 1000.0 * HISTORY_RETRIES, history=[UnreadableDataset()])

        self.assertIsNotNone(op.trained_day)
        self.assertFalse(self.needs_retraining_at(op, 1000.0 * HISTORY_RETRIES + 10 * HISTORY_RETRY_S))

    def test_a_history_that_cannot_be_requested_is_not_retried(self):
        # A missing deployment configuration does not go away by waiting.
        op = self.operator()
        self.train_at(op, 1000.0, history_error=RuntimeError("no deployment configuration"))

        self.assertIsNotNone(op.trained_day)
        self.assertFalse(self.needs_retraining_at(op, 1000.0 + 10 * HISTORY_RETRY_S))


class TestMerge(OperatorTestCase):
    def write_legacy(self, weekday=(), weekend=()):
        stored = collections.defaultdict(list)
        stored["weekday"] = [pd.Timestamp(w) for w in weekday]
        stored["weekend"] = [pd.Timestamp(w) for w in weekend]
        with open(os.path.join(self.data_path, "window_opening_times.pickle"), "wb") as f:
            pickle.dump(stored, f)

    def test_cold_merge_of_previous_model_recent_and_legacy_once(self):
        a, b, c = ns("2026-01-05 07:00:00"), ns("2026-01-06 07:00:00"), ns("2026-01-07 07:00:00")
        w = ns("2026-01-10 10:00:00")
        previous = OpeningsModel([a], [], trained_until=ns("2026-01-06 00:00:00"), legacy_imported=False)
        self.write_legacy(weekday=["2026-01-07 07:00:00", "2025-11-01 07:00:00"], weekend=["2026-01-10 10:00:00"])
        op = self.operator()
        op._ensure_loaded()
        op.recent = [a, b]

        model = self.train(op, model=FakePyFuncModel(previous), history_error=RuntimeError("down"))

        self.assertEqual(model.weekday, [a, b, c])
        self.assertEqual(model.weekend, [w])
        self.assertTrue(model.legacy_imported)
        self.assertEqual(model.trained_until, ns("2026-01-11 23:00:00"))

        self.write_legacy(weekday=["2026-01-08 07:00:00"])
        again = self.train(op, model=FakePyFuncModel(model), history_error=RuntimeError("down"))
        self.assertEqual(again.weekday, [a, b, c])

    def test_history_replaces_other_sources_where_it_reaches(self):
        openings = ["2026-01-05 07:00:00", "2026-01-06 07:00:00"]
        history = [FakeDataset(history_frame(openings, closings=["2025-12-20 12:00:00", "2026-01-05 07:30:00"]))]
        before_history = ns("2025-12-15 07:00:00")
        kafka_time_of_history_opening = ns("2026-01-05 07:00:00") + 300 * 10**6
        too_old = ns("2025-11-10 07:00:00")
        previous = OpeningsModel([before_history, kafka_time_of_history_opening, too_old], [],
                                 trained_until=ns("2026-01-06 00:00:00"), legacy_imported=True)
        op = self.operator(contact_sensor="")
        op._ensure_loaded()
        op.recent = [ns("2026-01-06 07:00:00") + 10**6]

        model = self.train(op, model=FakePyFuncModel(previous), history=history)

        self.assertEqual(model.weekday, [before_history] + [ns(o) for o in openings])

    def test_legacy_openings_before_the_history_are_kept(self):
        history = [FakeDataset(history_frame(["2026-01-05 07:00:00"], closings=["2026-01-01 12:00:00"]))]
        self.write_legacy(weekday=["2025-12-29 07:00:00", "2026-01-05 07:00:01"])
        op = self.operator(contact_sensor="")

        model = self.train(op, history=history)

        self.assertEqual(model.weekday, [ns("2025-12-29 07:00:00"), ns("2026-01-05 07:00:00")])
        self.assertTrue(model.legacy_imported)

    def test_unreadable_legacy_file_is_retried(self):
        with open(os.path.join(self.data_path, "window_opening_times.pickle"), "wb") as f:
            f.write(b"not a pickle")
        op = self.operator()
        model = self.train(op, history_error=RuntimeError("down"))
        self.assertFalse(model.legacy_imported)


class TestModelSerialisation(OperatorTestCase):
    def test_pickle_round_trip(self):
        model = OpeningsModel([3, 1, 2, 2], [5], trained_until=7, legacy_imported=True)
        loaded = pickle.loads(pickle.dumps(model))
        self.assertEqual((loaded.weekday, loaded.weekend, loaded.trained_until, loaded.legacy_imported),
                         ([1, 2, 3], [5], 7, True))

    def test_mlflow_round_trip_and_predict(self):
        model = OpeningsModel.from_openings([ns(f"2026-01-{d} 07:00:00") for d in ("05", "06", "07")],
                                            trained_until=ns("2026-01-08 00:00:00"))
        previous_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)
        mlflow.set_tracking_uri("file://" + os.path.join(self.data_path, "mlruns"))
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(artifact_path="m", python_model=model)
        loaded = mlflow.pyfunc.load_model(info.model_uri)

        unwrapped = loaded.unwrap_python_model()
        self.assertIsInstance(unwrapped, OpeningsModel)
        self.assertEqual((unwrapped.weekday, unwrapped.trained_until), (model.weekday, model.trained_until))
        self.assertEqual(loaded.predict([berlin("2026-01-12 00:00:00")]), [[{"start": "07:00:00", "end": "07:00:00"}]])
        self.assertEqual(loaded.predict([berlin("2026-01-10 00:00:00")]), [[]])


class TestState(OperatorTestCase):
    def test_restart_without_stop_keeps_recent_openings_and_the_last_day(self):
        op = self.operator()
        for day in ("12", "13", "14"):
            self.open_at(op, None, f"2026-01-{day} 07:00:00")

        restarted = self.operator()
        _, result = self.infer(restarted, None, "2026-01-15 00:01:00", window_open=True)
        self.assertEqual([r["stopping_time"] for r in result], ["2026-01-15T05:50:00Z"])

    def test_stop_saves_the_last_timestamp(self):
        op = self.operator()
        self.infer(op, None, "2026-01-12 07:00:00", window_open=False)
        self.infer(op, None, "2026-01-12 09:00:00", window_open=True)
        op.stop()
        with open(os.path.join(self.data_path, STATE_FILE)) as f:
            self.assertEqual(json.load(f)["last_timestamp"], ns("2026-01-12 09:00:00"))

    def test_last_timestamp_of_the_previous_version_is_used(self):
        with open(os.path.join(self.data_path, "last_timestamp.pickle"), "wb") as f:
            pickle.dump(pd.Timestamp("2026-01-12 22:00:00"), f)
        op = self.operator()
        self.infer(op, None, "2026-01-13 00:30:00", window_open=True)
        self.assertTrue(op.need_retraining(None))

    def test_training_takes_over_the_recent_openings_late_ones_stay(self):
        op = self.operator()
        self.open_at(op, None, "2026-01-12 07:00:00")
        model = self.train(op, history=[], now="2026-01-13 00:01:00")
        self.assertIn(ns("2026-01-12 07:00:00"), model.weekday)
        self.assertEqual(op.recent, [])

        # Arrives after the training although it is older: it must stay until the next training.
        self.open_at(op, FakePyFuncModel(model), "2026-01-12 23:58:00")
        self.open_at(op, FakePyFuncModel(model), "2026-01-13 00:05:00")
        self.assertEqual(op.recent, [ns("2026-01-12 23:58:00"), ns("2026-01-13 00:05:00")])

    def test_an_opening_is_saved_at_once_and_the_repeated_state_adds_none(self):
        op = self.operator()

        def saved():
            with open(os.path.join(self.data_path, STATE_FILE)) as f:
                return json.load(f)

        with mock.patch("main.time.monotonic", return_value=1000.0):
            self.open_at(op, None, "2026-01-12 07:00:00")
            self.infer(op, None, "2026-01-12 07:01:00", window_open=False)
            self.assertEqual(saved()["recent"], [ns("2026-01-12 07:00:00")])
            self.open_at(op, None, "2026-01-12 07:20:00")
            self.assertEqual(saved()["recent"], [ns("2026-01-12 07:00:00"), ns("2026-01-12 07:20:00")])
            self.assertTrue(saved()["window_open"])

    def test_the_first_message_after_a_cold_start_is_no_opening(self):
        op = self.operator()
        self.infer(op, None, "2026-01-12 07:00:00", window_open=False)
        self.infer(op, None, "2026-01-12 07:01:00", window_open=False)
        self.assertEqual(op.recent, [])

    def test_the_window_state_survives_a_restart(self):
        op = self.operator()
        self.infer(op, None, "2026-01-12 06:59:00", window_open=True)
        restarted = self.operator()
        self.infer(restarted, None, "2026-01-12 07:00:00", window_open=False)
        self.assertEqual(restarted.recent, [ns("2026-01-12 07:00:00")])

    def test_a_late_message_is_no_opening(self):
        op = self.operator()
        self.infer(op, None, "2026-01-12 07:00:00", window_open=True)
        self.infer(op, None, "2026-01-12 06:30:00", window_open=False)
        self.infer(op, None, "2026-01-12 07:01:00", window_open=True)
        self.assertEqual(op.recent, [])

class _LoggerMock:
    def __init__(self, *args, **kwargs):
        pass

    def trace(self, name):
        return contextlib.nullcontext()

    def log_metrics(self, metrics, step=None):
        pass


class TestLifecycle(OperatorTestCase):
    """init() trains when the registry holds no model, run() infers and retrains at the day change."""

    def test_init_trains_once_and_run_retrains_at_the_day_change(self):
        openings, closings = weekday_and_weekend_openings("2025-12-15", "2026-01-11")
        history = [FakeDataset(history_frame(openings, closings))]
        with open(os.path.join(self.data_path, "window_opening_times.pickle"), "wb") as f:
            pickle.dump(collections.defaultdict(list, {"weekday": [pd.Timestamp("2025-12-01 07:00:00")]}), f)

        registry = []

        def log_model(artifact_path, python_model):
            registry.append(pickle.loads(pickle.dumps(python_model)))
            return mock.MagicMock()

        def load_model(uri):
            if not registry:
                raise RuntimeError("no registered model")
            return FakePyFuncModel(registry[-1])

        op = Operator()
        with mock.patch("operator_lib.util.op_ml.ray"), \
             mock.patch("operator_lib.util.op_ml.mlflow") as mock_mlflow, \
             mock.patch("operator_lib.util.op_ml.MlflowClient"), \
             mock.patch("operator_lib.util.op_ml.TrainMlflowLogger", _LoggerMock), \
             training_without_ray(history):
            mock_mlflow.pyfunc.log_model.side_effect = log_model
            mock_mlflow.pyfunc.load_model.side_effect = load_model
            op.init(kafka_consumer=None, kafka_producer=None, filter_handler=None, output_topic=None,
                    pipeline_id="p", operator_id="o",
                    config=CustomConfig({"data_path": self.data_path, "contact_sensor": "",
                                         "training_end": "2026-01-11T22:00:00Z"}))
            self.assertEqual(len(registry), 1)
            self.assertIn(ns("2025-12-01 07:00:00"), registry[0].weekday)
            self.assertTrue(registry[0].legacy_imported)

            ts = berlin("2026-01-11 23:30:00").to_pydatetime()
            self.assertEqual(op.run({"window_open": False}, "s", "d", ts), (None, None))
            ts = berlin("2026-01-12 00:05:00").to_pydatetime()
            clock.set_fixed(ts)
            dt, result = op.run({"window_open": False}, "s", "d", ts)

        self.assertIsNone(dt)
        self.assertEqual(result, [{
            "stopping_time": "2026-01-12T05:50:00Z",
            "overall_confidence": "1.0",
            "timestamp": "2026-01-11T23:05:00Z",
        }])
        self.assertEqual(len(registry), 2)


if __name__ == '__main__':
    unittest.main()
