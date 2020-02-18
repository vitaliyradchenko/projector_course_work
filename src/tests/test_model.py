import json
from typing import List, Tuple

from ..predictor.news_predictor import NewsPredictor


class TestMainPredictor:
    _TEST_FILE_PATH = "src/tests/test_examples.json"
    _BASE_PATH = "model/"

    def test_predict(self) -> None:
        data, answers = self._load_data_and_answers()
        predictor = NewsPredictor.load(self._BASE_PATH)

        assert predictor.predict(data) == answers

    def _load_data_and_answers(self) -> Tuple[List[str], List[str]]:
        with open(self._TEST_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [sample["text"] for sample in data], [sample["answer"] for sample in data]
