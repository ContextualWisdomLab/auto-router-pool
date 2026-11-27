import json
import tempfile
import unittest
from pathlib import Path

from auto_router_pool import modality_benchmarks


class ModalityBenchmarkTest(unittest.TestCase):
    def test_pinned_scores_have_a_public_source(self):
        scores = modality_benchmarks.load()
        self.assertEqual(scores["image"]["gpt-image-2.5-sunburst"], 1230.0)
        self.assertIn("artificialanalysis.ai", modality_benchmarks.SOURCES["image"])
        self.assertIn("artificialanalysis.ai", modality_benchmarks.SOURCES["audio"])

    def test_home_file_overrides_and_adds_scores(self):
        with tempfile.TemporaryDirectory() as home:
            Path(home, "auto-pool-modality-benchmarks.json").write_text(
                json.dumps({
                    "image": {"new-image": 99},
                    "audio": {"gpt-transcribe": {"score": 99.5}},
                    "bogus": {"x": 1},
                })
            )
            scores = modality_benchmarks.load(Path(home))
        self.assertEqual(scores["image"]["new-image"], 99.0)
        self.assertEqual(scores["audio"]["gpt-transcribe"], 99.5)
        self.assertNotIn("bogus", scores)


if __name__ == "__main__":
    unittest.main()
