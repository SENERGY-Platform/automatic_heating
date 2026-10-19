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

import json
import os
import unittest

from operator_lib.manifest import build_manifest

from main import Operator

MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "operator-manifest.json")


class TestManifest(unittest.TestCase):
    def test_committed_manifest_matches_the_declarations(self):
        # The platform registers a version from the committed file; regenerate it with
        # `uv run python -m operator_lib manifest main.py` after changing a declaration.
        with open(MANIFEST) as f:
            committed = json.load(f)
        self.assertEqual(committed, build_manifest(Operator))


if __name__ == '__main__':
    unittest.main()
