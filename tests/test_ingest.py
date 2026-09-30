import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "ingest.py"
SPEC = importlib.util.spec_from_file_location("ingest", MODULE_PATH)
ingest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ingest)


class InferMetadataTests(unittest.TestCase):
    def test_10q_short_year(self):
        metadata = ingest.infer_metadata(Path("10Q_Q2_26.pdf"))
        self.assertEqual((metadata["document_type"], metadata["year"], metadata["quarter"]),
                         ("10-Q", 2026, "Q2"))

    def test_earnings_release_full_year(self):
        metadata = ingest.infer_metadata(Path("AMZN-Q3-2025-Earnings-Release.pdf"))
        self.assertEqual((metadata["document_type"], metadata["year"], metadata["quarter"]),
                         ("earnings_release", 2025, "Q3"))

    def test_compact_slide_period(self):
        metadata = ingest.infer_metadata(Path("Webslides_Q120_4.28.20_Final.pdf"))
        self.assertEqual((metadata["document_type"], metadata["year"], metadata["quarter"]),
                         ("earnings_slides", 2020, "Q1"))

    def test_generic_q4_2021_filename(self):
        metadata = ingest.infer_metadata(Path("business_and_financial_update.pdf"))
        self.assertEqual((metadata["document_type"], metadata["year"], metadata["quarter"]),
                         ("earnings_release", 2021, "Q4"))


if __name__ == "__main__":
    unittest.main()
