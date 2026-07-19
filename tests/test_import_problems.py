import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_problems import convert_evalplus_item, label_topic
from scripts.import_problems_probe import select_hard

FAKE_HE_ITEM = {
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements",
    "prompt": (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer '
        'to each other than given threshold. """\n'
    ),
    "canonical_solution": (
        "    for i, a in enumerate(numbers):\n"
        "        for j, b in enumerate(numbers):\n"
        "            if i != j and abs(a - b) < threshold:\n"
        "                return True\n"
        "    return False\n"
    ),
    "base_input": [[[1.0, 2.0, 3.0], 0.5], [[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3]],
}


class TestConvert:
    def test_maps_schema(self):
        p = convert_evalplus_item(FAKE_HE_ITEM, source="humaneval+")
        assert p["id"] == "h2_humaneval_0"
        assert p["function_name"] == "has_close_elements"
        assert p["source"] == "humaneval+"
        assert p["difficulty"] == "hard"
        assert len(p["tests"]) == 2
        assert p["tests"][0] == {"args": [[1.0, 2.0, 3.0], 0.5], "expected": False}
        assert p["tests"][1]["expected"] is True
        assert "closer to each other" in p["question"]
        assert p["gold_solution"].startswith("from typing import List")

    def test_rejects_non_json_io(self):
        bad = dict(FAKE_HE_ITEM, base_input=[[{1, 2, 3}]])  # set: not JSON-safe
        assert convert_evalplus_item(bad, source="humaneval+") is None


class TestLabelTopic:
    def test_keywords(self):
        assert label_topic("find the longest common subsequence", "") == "dp"
        assert label_topic("reverse the given string", "") == "strings"
        assert label_topic("shortest path between nodes in a graph", "") == "graphs"
        assert label_topic("compute the sum of two numbers", "") == "arithmetic"
        assert label_topic("do the thing", "") == "arrays"


def _cand(i, topic):
    return {
        "id": f"h2_c{i}",
        "question": "q",
        "function_name": "f",
        "tests": [],
        "topic": topic,
        "difficulty": "hard",
        "gold_solution": "def f(): pass",
        "source": "mbpp+",
    }


class TestSelectHard:
    def test_filters_by_pass_rate(self):
        results = [
            (_cand(1, "dp"), 0.0),
            (_cand(2, "dp"), 0.5),
            (_cand(3, "dp"), 1.0),
        ]
        kept = select_hard(results, max_keep=10)
        assert [c["id"] for c in kept] == ["h2_c1", "h2_c2"]

    def test_topic_balance_under_cap(self):
        results = [(_cand(i, "dp"), 0.0) for i in range(10)]
        results += [(_cand(100 + i, "graphs"), 0.0) for i in range(2)]
        kept = select_hard(results, max_keep=6)
        topics = [c["topic"] for c in kept]
        assert len(kept) == 6
        assert topics.count("graphs") == 2  # scarce topic fully kept
        assert topics.count("dp") == 4
