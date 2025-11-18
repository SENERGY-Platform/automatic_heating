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

from operator_lib.util import OperatorBase, logger, InitPhase
from operator_lib.util.persistence import save, load
import os
import datetime
import pandas as pd
from algo import compute_clusters_boundaries, compute_confidence_from_spreading, compute_confidence_by_daily_apperance
from collections import defaultdict



FIRST_DATA_FILENAME = "first_data_time.pickle"
LAST_TIMESTAMP_FILE = "last_timestamp.pickle"
WINDOW_OPENING_TIMES_FILE = "window_opening_times.pickle"


from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 28
    init_phase_level: str = "d"
    
    high_confidence_boundary: int = 600 # in seconds, corresponds to 10 minutes
    low_confidence_boundary: int = 3600 # in seconds, corresponds to 1 hour

    inertia_buffer: int = 10 # in minutes

    confidence_days: int = 7 # in days

    contact_sensor: bool = True

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)

        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 28
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

class Operator(OperatorBase):
    configType = CustomConfig

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.high_confidence_boundary = int(self.config.high_confidence_boundary)
        self.low_confidence_boundary = int(self.config.low_confidence_boundary)

        self.inertia_buffer = pd.Timedelta(int(self.config.inertia_buffer), "min")

        self.confidence_days = int(self.config.confidence_days)

        self.contact_sensor = bool(self.config.contact_sensor)

        self.first_data_time = load(self.data_path, FIRST_DATA_FILENAME)

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
            "stopping_time": pd.Timestamp.now().isoformat(),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self.init_phase_handler.send_first_init_msg(value)

        self.last_timestamp = load(self.data_path, LAST_TIMESTAMP_FILE)
        
    def stop(self):
        super().stop()
        save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
        save(self.data_path, LAST_TIMESTAMP_FILE, self.last_timestamp)

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        # Convert to german time and then forget the timezone.
        current_timestamp = pd.Timestamp(timestamp).tz_localize("Zulu").tz_convert("Europe/Berlin").tz_localize(None)

        if self.first_data_time == None:
            self.first_data_time = current_timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        if self.last_timestamp == None:
            self.last_timestamp = current_timestamp

        weekend = self.check_if_weekend(current_timestamp)

        real_time_data = (pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None) - current_timestamp < pd.Timedelta(10, "s"))

        if self.contact_sensor:
            window_open = not bool(data["window_open"])
        else:
            window_open = bool(data["window_open"]) 

        if real_time_data:
            logger.debug(f"{current_timestamp}:  Window open: {window_open}!")
        else:
            logger.debug(f"Historic data from: {current_timestamp}:  Window open: {window_open}!")


        if window_open:
            window_opening_times = load(self.data_path, WINDOW_OPENING_TIMES_FILE, default=defaultdict(list))
            if weekend:
                window_opening_times["weekend"].append(current_timestamp)
            else:
                window_opening_times["weekday"].append(current_timestamp)
            
            # If data from more than 60 days is stored delete entries.
            window_opening_times["weekend"] = [ts for ts in window_opening_times["weekend"] if current_timestamp - ts <= pd.Timedelta(60,"d")]
            window_opening_times["weekday"] = [ts for ts in window_opening_times["weekday"] if current_timestamp - ts <= pd.Timedelta(60,"d")]


            save(self.data_path, WINDOW_OPENING_TIMES_FILE, window_opening_times)
            # window_opening_times is a potentially growing list->it's better to not hold it inside the memory all the time
            del window_opening_times 
        
        outcome = self.check_for_init_phase(current_timestamp)
        if outcome: return outcome  # init phase cuts of normal analysis

        new_day = self.check_for_new_day(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp


        confidence_list = []
        if new_day:
            window_opening_times = load(self.data_path, WINDOW_OPENING_TIMES_FILE, default=defaultdict(list))
            if weekend:
                considered_timestamps = window_opening_times["weekend"]
            else:
                considered_timestamps = window_opening_times["weekday"]
            
            if len(considered_timestamps) <= 2:
                logger.debug({"stopping_time": "Not enough data!",
                                        "confidence_by_spreading": str(0),
                                        "confidence_by_dailyappearance": str(0),
                                        "overall_confidence": str(0),
                                        "timestamp": current_timestamp.isoformat()})
            else:
                clusters_boundaries, indices = compute_clusters_boundaries(considered_timestamps)
                current_day = current_timestamp.floor("d")
                for c in clusters_boundaries.keys():
                    pair_of_boundaries = clusters_boundaries[c]
                    ts_in_cluster = [considered_timestamps[i] for i in indices[c]]
                    confidence_by_spreading = compute_confidence_from_spreading(ts_in_cluster, self.high_confidence_boundary, self.low_confidence_boundary)
                    confidence_by_daily_appearance = compute_confidence_by_daily_apperance(current_timestamp, considered_timestamps, pair_of_boundaries, confidence_days=self.confidence_days)
                    overall_confidence = confidence_by_spreading * confidence_by_daily_appearance
                
                    confidence_list.append({"stopping_time": (pd.Timestamp.combine(current_day, pair_of_boundaries[0]) - self.inertia_buffer).isoformat(),
                                            "confidence_by_spreading": str(confidence_by_spreading),
                                            "confidence by daily_ appearance": str(confidence_by_daily_appearance),
                                            "overall_confidence": str(overall_confidence),
                                            "timestamp": current_timestamp.isoformat()})
                del window_opening_times
                logger.debug(f"Results for next day: {confidence_list}")
                return [{key: confidence_entry[key] for key in ["stopping_time", "overall_confidence", "timestamp"]} for confidence_entry in confidence_list]
            




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
        
    def check_for_init_phase(self, current_timestamp: pd.Timestamp):
        init_value = {
            "stopping_time": current_timestamp.isoformat(),
            "overall_confidence": None,
            "timestamp": current_timestamp.isoformat()
        }
        if self.init_phase_handler.operator_is_in_init_phase(current_timestamp):
            logger.debug(self.init_phase_handler.generate_init_msg(current_timestamp, init_value))
            return self.init_phase_handler.generate_init_msg(current_timestamp, init_value)

        if self.init_phase_handler.init_phase_needs_to_be_reset():
            logger.debug(self.init_phase_handler.reset_init_phase(init_value))
            return self.init_phase_handler.reset_init_phase(init_value)

        return False

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="open-window-detection-operator", git_info_file='git_commit')
