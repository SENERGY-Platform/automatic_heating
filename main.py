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

from operator_lib.util import OperatorBase, logger, InitPhase, todatetime, timestamp_to_str
from operator_lib.util.persistence import save, load
import os
import datetime
import pandas as pd
from algo import compute_clusters_boundaries, compute_confidence_from_spreading, compute_confidence_by_daily_apperance



# TODO: Logging!


FIRST_DATA_FILENAME = "first_data_time.pickle"
LAST_TIMESTAMP_FILE = "last_timestamp.pickle"
WINDOW_OPENING_TIMES_FILE = "window_opening_times.pickle"

X_DAYS = 7
HIGH_CONFIDENCE_BOUNDARY = 600 # in seconds, corresponds to 10 minutes
LOW_CONFIDENCE_BOUNDARY = 3600 # in seconds, corresponds to 1 hour

INERTIA_BUFFER = pd.Timedelta(10, "min")

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"
    notification_mode: str = "never"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)

        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

class Operator(OperatorBase):
    configType = CustomConfig

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.first_data_time = load(self.data_path, FIRST_DATA_FILENAME)

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
            "timestamp": timestamp_to_str(pd.Timestamp.now())
        }
        self.init_phase_handler.send_first_init_msg(value)

        self.last_timestamp = load(self.data_path, LAST_TIMESTAMP_FILE)
        #self.window_opening_times = load(self.data_path, WINDOW_OPENING_TIMES_FILE, default=[])
        
    def stop(self):
        super().stop()
        save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
        save(self.data_path, LAST_TIMESTAMP_FILE, self.last_timestamp)

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        current_timestamp = pd.Timestamp(timestamp)
        # Discard data, which is too old.
        if pd.Timestamp.now() - current_timestamp > pd.Timedelta(30, "d"):
            return


        if self.first_data_time == None:
            self.first_data_time = current_timestamp
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        if self.last_timestamp == None:
            self.last_timestamp = current_timestamp

        real_time_data = (pd.Timestamp.now() - current_timestamp < pd.Timedelta(10, "s"))

        window_open = not bool(data["window_open"])

        if real_time_data:
            logger.debug(f"{current_timestamp}:  Window open: {window_open}!")
        else:
            logger.debug(f"Historic data from: {current_timestamp}:  Window open: {window_open}!")


        if window_open:
            self.window_opening_times = load(self.data_path, WINDOW_OPENING_TIMES_FILE, default=[])
            self.window_opening_times.append(current_timestamp)
            save(self.data_path, WINDOW_OPENING_TIMES_FILE, self.window_opening_times)
            # self.window_opening_times is a potentially growing list->it's better to not hold it inside the memory all the time
            del self.window_opening_times 
        #elif window_open and not real_time_data:
            # If historic data is loaded we should hold self.window_opening_times inside memory to avoid saving and loading data all the time.
            #self.window_opening_times.append(current_timestamp)
        
        outcome = self.check_for_init_phase(current_timestamp)
        if outcome: return outcome  # init phase cuts of normal analysis

        new_day = self.check_for_new_day(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp


        confidence_list = []
        if new_day:
            #if real_time_data:
            self.window_opening_times = load(self.data_path, WINDOW_OPENING_TIMES_FILE, default=[])
            clusters_boundaries = compute_clusters_boundaries(self.window_opening_times)
            current_day = current_timestamp.floor("d")
            for c in clusters_boundaries.keys():
                pair_of_boundaries = clusters_boundaries[c]

                confidence_by_spreading = compute_confidence_from_spreading(self.window_opening_times, HIGH_CONFIDENCE_BOUNDARY, LOW_CONFIDENCE_BOUNDARY)
                confidence_by_daily_appearance = compute_confidence_by_daily_apperance(self.window_opening_times, pair_of_boundaries, x_days=X_DAYS)

                overall_confidence = confidence_by_spreading * confidence_by_daily_appearance
                
                confidence_list.append({"stopping_time": timestamp_to_str(pd.Timestamp.combine(current_day, pair_of_boundaries[0]) - INERTIA_BUFFER),
                                        "overall_confidence": str(overall_confidence),
                                        "timestamp": timestamp_to_str(current_timestamp)})
            logger.debug(f"Results for next day: {confidence_list}")
            del self.window_opening_times
            return confidence_list





    def check_for_new_day(self, last_timestamp: pd.Timestamp, current_timestamp: pd.Timestamp):
        if current_timestamp.date() > last_timestamp.date():
            return True
        else:
            return False
        
    def check_for_init_phase(self, current_timestamp):
        init_value = {
            "stopping_time": None,
            "overall_confidence": None,
            "timestamp": timestamp_to_str(current_timestamp)
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
