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

import unittest

import pandas as pd

from main import Operator


class TestPrepareOutputTimestamp(unittest.TestCase):
    def convert(self, wall_time):
        return Operator.prepare_output_timestamp(None, pd.Timestamp(wall_time))

    def test_summer_and_winter_time(self):
        self.assertEqual(self.convert("2026-07-01 12:00:00"), "2026-07-01T10:00:00Z")
        self.assertEqual(self.convert("2026-01-01 12:00:00"), "2026-01-01T11:00:00Z")

    def test_time_in_the_spring_gap_moves_to_three(self):
        self.assertEqual(self.convert("2026-03-29 02:30:49.691"), "2026-03-29T01:00:00Z")

    def test_repeated_autumn_hour_takes_standard_time(self):
        self.assertEqual(self.convert("2026-10-25 02:30:00"), "2026-10-25T01:30:00Z")


if __name__ == '__main__':
    unittest.main()
