#!/usr/bin/env python3
"""Generate fixtures/coding_subset.json — easy/hard coding problems with unit tests.

Run from repo root:
  python3 scripts/generate_coding_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "fixtures" / "coding_subset.json"


def P(
    id: str,
    question: str,
    function_name: str,
    tests: list,
    topic: str,
    difficulty: str,
    gold: str,
) -> dict:
    return {
        "id": id,
        "question": question.strip(),
        "function_name": function_name,
        "tests": tests,
        "topic": topic,
        "difficulty": difficulty,
        "gold_solution": gold.strip(),
    }


PROBLEMS: list[dict] = []

# ---------------------------------------------------------------------------
# EASY — arrays / strings / arithmetic (baseline)
# ---------------------------------------------------------------------------

PROBLEMS += [
    P(
        "e_sum_two",
        "Write a function add(a, b) that returns the sum of two integers.",
        "add",
        [{"args": [1, 2], "expected": 3}, {"args": [-1, 5], "expected": 4}, {"args": [0, 0], "expected": 0}],
        "arithmetic",
        "easy",
        "def add(a, b):\n    return a + b",
    ),
    P(
        "e_max_two",
        "Write maximum(a, b) returning the larger of two numbers.",
        "maximum",
        [{"args": [3, 7], "expected": 7}, {"args": [9, 2], "expected": 9}, {"args": [5, 5], "expected": 5}],
        "arithmetic",
        "easy",
        "def maximum(a, b):\n    return a if a >= b else b",
    ),
    P(
        "e_abs_val",
        "Write abs_val(x) returning the absolute value of x.",
        "abs_val",
        [{"args": [-3], "expected": 3}, {"args": [4], "expected": 4}, {"args": [0], "expected": 0}],
        "arithmetic",
        "easy",
        "def abs_val(x):\n    return -x if x < 0 else x",
    ),
    P(
        "e_is_even",
        "Write is_even(n) returning True if n is even else False.",
        "is_even",
        [{"args": [2], "expected": True}, {"args": [7], "expected": False}, {"args": [0], "expected": True}],
        "arithmetic",
        "easy",
        "def is_even(n):\n    return n % 2 == 0",
    ),
    P(
        "e_factorial",
        "Write factorial(n) for n >= 0 (factorial(0)=1).",
        "factorial",
        [{"args": [0], "expected": 1}, {"args": [1], "expected": 1}, {"args": [5], "expected": 120}],
        "arithmetic",
        "easy",
        "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
    ),
    P(
        "e_sum_list",
        "Write sum_list(nums) returning the sum of a list of integers. Empty list -> 0.",
        "sum_list",
        [{"args": [[1, 2, 3]], "expected": 6}, {"args": [[]], "expected": 0}, {"args": [[-1, 1]], "expected": 0}],
        "arrays",
        "easy",
        "def sum_list(nums):\n    return sum(nums)",
    ),
    P(
        "e_count_pos",
        "Write count_positive(nums) counting how many elements are > 0.",
        "count_positive",
        [{"args": [[1, -2, 3, 0]], "expected": 2}, {"args": [[-1, -2]], "expected": 0}, {"args": [[5]], "expected": 1}],
        "arrays",
        "easy",
        "def count_positive(nums):\n    return sum(1 for x in nums if x > 0)",
    ),
    P(
        "e_reverse_list",
        "Write reverse_list(nums) returning a new list that is the reverse of nums.",
        "reverse_list",
        [{"args": [[1, 2, 3]], "expected": [3, 2, 1]}, {"args": [[]], "expected": []}, {"args": [[7]], "expected": [7]}],
        "arrays",
        "easy",
        "def reverse_list(nums):\n    return nums[::-1]",
    ),
    P(
        "e_contains",
        "Write contains(nums, target) returning True if target is in nums.",
        "contains",
        [{"args": [[1, 2, 3], 2], "expected": True}, {"args": [[1, 2], 9], "expected": False}, {"args": [[], 1], "expected": False}],
        "arrays",
        "easy",
        "def contains(nums, target):\n    return target in nums",
    ),
    P(
        "e_first_last",
        "Write first_last(nums) returning [first, last]. Assume nums is non-empty.",
        "first_last",
        [{"args": [[4, 5, 6]], "expected": [4, 6]}, {"args": [[9]], "expected": [9, 9]}],
        "arrays",
        "easy",
        "def first_last(nums):\n    return [nums[0], nums[-1]]",
    ),
    P(
        "e_strlen",
        "Write strlen(s) returning the length of string s.",
        "strlen",
        [{"args": ["hi"], "expected": 2}, {"args": [""], "expected": 0}, {"args": ["abc"], "expected": 3}],
        "strings",
        "easy",
        "def strlen(s):\n    return len(s)",
    ),
    P(
        "e_upper",
        "Write to_upper(s) returning s in uppercase.",
        "to_upper",
        [{"args": ["Hi"], "expected": "HI"}, {"args": ["a"], "expected": "A"}, {"args": [""], "expected": ""}],
        "strings",
        "easy",
        "def to_upper(s):\n    return s.upper()",
    ),
    P(
        "e_is_palindrome",
        "Write is_palindrome(s) True if s reads the same forwards and backwards (case-sensitive).",
        "is_palindrome",
        [{"args": ["aba"], "expected": True}, {"args": ["ab"], "expected": False}, {"args": [""], "expected": True}],
        "strings",
        "easy",
        "def is_palindrome(s):\n    return s == s[::-1]",
    ),
    P(
        "e_count_char",
        "Write count_char(s, c) counting occurrences of character c in s.",
        "count_char",
        [{"args": ["banana", "a"], "expected": 3}, {"args": ["hi", "z"], "expected": 0}, {"args": ["", "a"], "expected": 0}],
        "strings",
        "easy",
        "def count_char(s, c):\n    return s.count(c)",
    ),
    P(
        "e_join_words",
        "Write join_words(words) joining a list of words with a single space.",
        "join_words",
        [{"args": [["a", "b"]], "expected": "a b"}, {"args": [["hi"]], "expected": "hi"}, {"args": [[]], "expected": ""}],
        "strings",
        "easy",
        "def join_words(words):\n    return ' '.join(words)",
    ),
    P(
        "e_min_list",
        "Write min_list(nums) returning the minimum. Assume non-empty.",
        "min_list",
        [{"args": [[3, 1, 2]], "expected": 1}, {"args": [[-5, 0]], "expected": -5}],
        "arrays",
        "easy",
        "def min_list(nums):\n    return min(nums)",
    ),
    P(
        "e_product",
        "Write product(nums) returning the product of all elements. Empty -> 1.",
        "product",
        [{"args": [[2, 3, 4]], "expected": 24}, {"args": [[]], "expected": 1}, {"args": [[0, 5]], "expected": 0}],
        "arrays",
        "easy",
        "def product(nums):\n    r = 1\n    for x in nums:\n        r *= x\n    return r",
    ),
    P(
        "e_clamp",
        "Write clamp(x, lo, hi) returning x constrained to [lo, hi].",
        "clamp",
        [{"args": [5, 0, 10], "expected": 5}, {"args": [-1, 0, 10], "expected": 0}, {"args": [99, 0, 10], "expected": 10}],
        "arithmetic",
        "easy",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    ),
    P(
        "e_mean",
        "Write mean(nums) returning the arithmetic mean as float. Assume non-empty.",
        "mean",
        [{"args": [[2, 4]], "expected": 3.0}, {"args": [[1, 2, 3]], "expected": 2.0}],
        "arithmetic",
        "easy",
        "def mean(nums):\n    return sum(nums) / len(nums)",
    ),
    P(
        "e_fizz",
        "Write fizz(n): return 'fizz' if n divisible by 3 else str(n).",
        "fizz",
        [{"args": [3], "expected": "fizz"}, {"args": [4], "expected": "4"}, {"args": [9], "expected": "fizz"}],
        "arithmetic",
        "easy",
        "def fizz(n):\n    return 'fizz' if n % 3 == 0 else str(n)",
    ),
]

# more easy variants for baseline volume
for i, (a, b) in enumerate([(10, 3), (20, 7), (100, 9), (8, 2), (15, 4)]):
    PROBLEMS.append(
        P(
            f"e_mod_{i}",
            f"Write mod_ab(x, y) returning x % y. (Example focus: {a}, {b})",
            "mod_ab",
            [{"args": [a, b], "expected": a % b}, {"args": [7, 3], "expected": 1}, {"args": [0, 5], "expected": 0}],
            "arithmetic",
            "easy",
            "def mod_ab(x, y):\n    return x % y",
        )
    )

for i, s in enumerate(["hello", "world", "code", "test", "agent"]):
    PROBLEMS.append(
        P(
            f"e_revstr_{i}",
            f"Write reverse_str(s) reversing the string. Example input includes '{s}'.",
            "reverse_str",
            [{"args": [s], "expected": s[::-1]}, {"args": ["ab"], "expected": "ba"}, {"args": [""], "expected": ""}],
            "strings",
            "easy",
            "def reverse_str(s):\n    return s[::-1]",
        )
    )

for i, arr in enumerate([[1, 2], [5, 5, 5], [9, 1, 9], [0], [2, 0, 2]]):
    PROBLEMS.append(
        P(
            f"e_all_eq_{i}",
            "Write all_equal(nums) True if every element equals the first (empty True).",
            "all_equal",
            [
                {"args": [arr], "expected": len(set(arr)) <= 1},
                {"args": [[]], "expected": True},
                {"args": [[1, 2]], "expected": False},
            ],
            "arrays",
            "easy",
            "def all_equal(nums):\n    return all(x == nums[0] for x in nums) if nums else True",
        )
    )

# ---------------------------------------------------------------------------
# HARD — DP / graphs-ish / tricky logic (degraded + recovery)
# ---------------------------------------------------------------------------

PROBLEMS += [
    P(
        "h_two_sum_indices",
        "Write two_sum(nums, target) returning a list [i, j] of distinct indices such that "
        "nums[i] + nums[j] == target. Exactly one solution exists. Prefer the lexicographically "
        "smallest [i, j] if multiple.",
        "two_sum",
        [
            {"args": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"args": [[3, 2, 4], 6], "expected": [1, 2]},
            {"args": [[3, 3], 6], "expected": [0, 1]},
        ],
        "arrays",
        "hard",
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        if target - x in seen:\n"
        "            return [seen[target - x], i]\n"
        "        seen[x] = i\n"
        "    return []",
    ),
    P(
        "h_max_subarray",
        "Write max_subarray(nums) returning the maximum contiguous subarray sum (Kadane). "
        "nums is non-empty.",
        "max_subarray",
        [
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[1]], "expected": 1},
            {"args": [[-1, -2]], "expected": -1},
        ],
        "dp",
        "hard",
        "def max_subarray(nums):\n"
        "    best = cur = nums[0]\n"
        "    for x in nums[1:]:\n"
        "        cur = max(x, cur + x)\n"
        "        best = max(best, cur)\n"
        "    return best",
    ),
    P(
        "h_climb_stairs",
        "Write climb_stairs(n): number of distinct ways to climb n stairs taking 1 or 2 steps.",
        "climb_stairs",
        [{"args": [2], "expected": 2}, {"args": [3], "expected": 3}, {"args": [5], "expected": 8}],
        "dp",
        "hard",
        "def climb_stairs(n):\n"
        "    if n <= 2:\n"
        "        return n\n"
        "    a, b = 1, 2\n"
        "    for _ in range(3, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b",
    ),
    P(
        "h_coin_change",
        "Write coin_change(coins, amount): fewest coins to make amount, or -1 if impossible. "
        "coins are positive integers.",
        "coin_change",
        [
            {"args": [[1, 2, 5], 11], "expected": 3},
            {"args": [[2], 3], "expected": -1},
            {"args": [[1], 0], "expected": 0},
        ],
        "dp",
        "hard",
        "def coin_change(coins, amount):\n"
        "    INF = amount + 1\n"
        "    dp = [0] + [INF] * amount\n"
        "    for a in range(1, amount + 1):\n"
        "        for c in coins:\n"
        "            if c <= a:\n"
        "                dp[a] = min(dp[a], dp[a - c] + 1)\n"
        "    return dp[amount] if dp[amount] != INF else -1",
    ),
    P(
        "h_lis_length",
        "Write lis_length(nums): length of the longest strictly increasing subsequence.",
        "lis_length",
        [
            {"args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expected": 4},
            {"args": [[0, 1, 0, 3, 2, 3]], "expected": 4},
            {"args": [[7, 7, 7]], "expected": 1},
        ],
        "dp",
        "hard",
        "def lis_length(nums):\n"
        "    if not nums:\n"
        "        return 0\n"
        "    dp = [1] * len(nums)\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i):\n"
        "            if nums[j] < nums[i]:\n"
        "                dp[i] = max(dp[i], dp[j] + 1)\n"
        "    return max(dp)",
    ),
    P(
        "h_unique_paths",
        "Write unique_paths(m, n): paths from top-left to bottom-right of an m x n grid "
        "moving only right or down.",
        "unique_paths",
        [{"args": [3, 7], "expected": 28}, {"args": [3, 2], "expected": 3}, {"args": [1, 1], "expected": 1}],
        "dp",
        "hard",
        "def unique_paths(m, n):\n"
        "    dp = [1] * n\n"
        "    for _ in range(1, m):\n"
        "        for j in range(1, n):\n"
        "            dp[j] += dp[j - 1]\n"
        "    return dp[-1]",
    ),
    P(
        "h_house_robber",
        "Write rob(nums): max money robbing houses in a line without adjacent houses.",
        "rob",
        [
            {"args": [[1, 2, 3, 1]], "expected": 4},
            {"args": [[2, 7, 9, 3, 1]], "expected": 12},
            {"args": [[2, 1, 1, 2]], "expected": 4},
        ],
        "dp",
        "hard",
        "def rob(nums):\n"
        "    prev2 = prev1 = 0\n"
        "    for x in nums:\n"
        "        prev2, prev1 = prev1, max(prev1, prev2 + x)\n"
        "    return prev1",
    ),
    P(
        "h_word_break",
        "Write word_break(s, word_dict): True if s can be segmented into a space-separated "
        "sequence of dictionary words (reuse allowed).",
        "word_break",
        [
            {"args": ["leetcode", ["leet", "code"]], "expected": True},
            {"args": ["applepenapple", ["apple", "pen"]], "expected": True},
            {"args": ["catsandog", ["cats", "dog", "sand", "and", "cat"]], "expected": False},
        ],
        "dp",
        "hard",
        "def word_break(s, word_dict):\n"
        "    words = set(word_dict)\n"
        "    n = len(s)\n"
        "    ok = [False] * (n + 1)\n"
        "    ok[0] = True\n"
        "    for i in range(1, n + 1):\n"
        "        for j in range(i):\n"
        "            if ok[j] and s[j:i] in words:\n"
        "                ok[i] = True\n"
        "                break\n"
        "    return ok[n]",
    ),
    P(
        "h_min_path_sum",
        "Write min_path_sum(grid): minimum path sum from top-left to bottom-right moving "
        "only right/down. grid is a non-empty list of equal-length lists.",
        "min_path_sum",
        [
            {"args": [[[1, 3, 1], [1, 5, 1], [4, 2, 1]]], "expected": 7},
            {"args": [[[1, 2, 3], [4, 5, 6]]], "expected": 12},
        ],
        "dp",
        "hard",
        "def min_path_sum(grid):\n"
        "    m, n = len(grid), len(grid[0])\n"
        "    dp = [0] * n\n"
        "    for i in range(m):\n"
        "        for j in range(n):\n"
        "            if i == 0 and j == 0:\n"
        "                dp[j] = grid[i][j]\n"
        "            elif i == 0:\n"
        "                dp[j] = dp[j - 1] + grid[i][j]\n"
        "            elif j == 0:\n"
        "                dp[j] = dp[j] + grid[i][j]\n"
        "            else:\n"
        "                dp[j] = min(dp[j], dp[j - 1]) + grid[i][j]\n"
        "    return dp[-1]",
    ),
    P(
        "h_can_jump",
        "Write can_jump(nums): True if you can reach the last index starting at 0, "
        "where nums[i] is max jump length from i.",
        "can_jump",
        [
            {"args": [[2, 3, 1, 1, 4]], "expected": True},
            {"args": [[3, 2, 1, 0, 4]], "expected": False},
            {"args": [[0]], "expected": True},
        ],
        "greedy",
        "hard",
        "def can_jump(nums):\n"
        "    reach = 0\n"
        "    for i, x in enumerate(nums):\n"
        "        if i > reach:\n"
        "            return False\n"
        "        reach = max(reach, i + x)\n"
        "    return True",
    ),
    P(
        "h_jump_game_ii",
        "Write jump(nums): minimum jumps to reach last index. Guaranteed reachable.",
        "jump",
        [
            {"args": [[2, 3, 1, 1, 4]], "expected": 2},
            {"args": [[2, 3, 0, 1, 4]], "expected": 2},
            {"args": [[1]], "expected": 0},
        ],
        "greedy",
        "hard",
        "def jump(nums):\n"
        "    jumps = end = farthest = 0\n"
        "    for i in range(len(nums) - 1):\n"
        "        farthest = max(farthest, i + nums[i])\n"
        "        if i == end:\n"
        "            jumps += 1\n"
        "            end = farthest\n"
        "    return jumps",
    ),
    P(
        "h_merge_intervals",
        "Write merge_intervals(intervals): merge overlapping [start, end] intervals, "
        "return sorted merged list.",
        "merge_intervals",
        [
            {"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expected": [[1, 6], [8, 10], [15, 18]]},
            {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
            {"args": [[[1, 4], [0, 4]]], "expected": [[0, 4]]},
        ],
        "arrays",
        "hard",
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    intervals = sorted(intervals)\n"
        "    out = [intervals[0][:]]\n"
        "    for s, e in intervals[1:]:\n"
        "        if s <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n"
        "            out.append([s, e])\n"
        "    return out",
    ),
    P(
        "h_product_except_self",
        "Write product_except_self(nums): for each index i, product of all elements except "
        "nums[i], without using division. Return a list.",
        "product_except_self",
        [
            {"args": [[1, 2, 3, 4]], "expected": [24, 12, 8, 6]},
            {"args": [[-1, 1, 0, -3, 3]], "expected": [0, 0, 9, 0, 0]},
        ],
        "arrays",
        "hard",
        "def product_except_self(nums):\n"
        "    n = len(nums)\n"
        "    out = [1] * n\n"
        "    left = 1\n"
        "    for i in range(n):\n"
        "        out[i] = left\n"
        "        left *= nums[i]\n"
        "    right = 1\n"
        "    for i in range(n - 1, -1, -1):\n"
        "        out[i] *= right\n"
        "        right *= nums[i]\n"
        "    return out",
    ),
    P(
        "h_group_anagrams_count",
        "Write group_anagrams_count(strs): return the number of anagram groups "
        "(same letters ignoring order).",
        "group_anagrams_count",
        [
            {"args": [["eat", "tea", "tan", "ate", "nat", "bat"]], "expected": 3},
            {"args": [[""]], "expected": 1},
            {"args": [["a"]], "expected": 1},
        ],
        "strings",
        "hard",
        "def group_anagrams_count(strs):\n"
        "    from collections import defaultdict\n"
        "    g = defaultdict(int)\n"
        "    for s in strs:\n"
        "        g[''.join(sorted(s))] += 1\n"
        "    return len(g)",
    ),
    P(
        "h_longest_common_subseq",
        "Write lcs_length(a, b): length of longest common subsequence of strings a and b.",
        "lcs_length",
        [
            {"args": ["abcde", "ace"], "expected": 3},
            {"args": ["abc", "abc"], "expected": 3},
            {"args": ["abc", "def"], "expected": 0},
        ],
        "dp",
        "hard",
        "def lcs_length(a, b):\n"
        "    m, n = len(a), len(b)\n"
        "    dp = [0] * (n + 1)\n"
        "    for i in range(1, m + 1):\n"
        "        prev = 0\n"
        "        for j in range(1, n + 1):\n"
        "            cur = dp[j]\n"
        "            if a[i - 1] == b[j - 1]:\n"
        "                dp[j] = prev + 1\n"
        "            else:\n"
        "                dp[j] = max(dp[j], dp[j - 1])\n"
        "            prev = cur\n"
        "    return dp[n]",
    ),
    P(
        "h_edit_distance",
        "Write edit_distance(a, b): Levenshtein edit distance between a and b.",
        "edit_distance",
        [
            {"args": ["horse", "ros"], "expected": 3},
            {"args": ["intention", "execution"], "expected": 5},
            {"args": ["", "a"], "expected": 1},
        ],
        "dp",
        "hard",
        "def edit_distance(a, b):\n"
        "    m, n = len(a), len(b)\n"
        "    dp = list(range(n + 1))\n"
        "    for i in range(1, m + 1):\n"
        "        prev = dp[0]\n"
        "        dp[0] = i\n"
        "        for j in range(1, n + 1):\n"
        "            cur = dp[j]\n"
        "            if a[i - 1] == b[j - 1]:\n"
        "                dp[j] = prev\n"
        "            else:\n"
        "                dp[j] = 1 + min(prev, dp[j], dp[j - 1])\n"
        "            prev = cur\n"
        "    return dp[n]",
    ),
    P(
        "h_num_islands",
        "Write num_islands(grid): count islands of '1' in a 2D grid of '1'/'0' "
        "(4-directional connectivity).",
        "num_islands",
        [
            {
                "args": [[["1", "1", "0"], ["1", "0", "0"], ["0", "0", "1"]]],
                "expected": 2,
            },
            {"args": [[["1", "0"], ["0", "1"]]], "expected": 2},
            {"args": [[["0"]]], "expected": 0},
        ],
        "graphs",
        "hard",
        "def num_islands(grid):\n"
        "    if not grid:\n"
        "        return 0\n"
        "    m, n = len(grid), len(grid[0])\n"
        "    def dfs(i, j):\n"
        "        if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] != '1':\n"
        "            return\n"
        "        grid[i][j] = '0'\n"
        "        dfs(i + 1, j); dfs(i - 1, j); dfs(i, j + 1); dfs(i, j - 1)\n"
        "    count = 0\n"
        "    for i in range(m):\n"
        "        for j in range(n):\n"
        "            if grid[i][j] == '1':\n"
        "                count += 1\n"
        "                dfs(i, j)\n"
        "    return count",
    ),
    P(
        "h_course_order_ok",
        "Write can_finish(num_courses, prerequisites): True if you can finish all courses. "
        "prerequisites is a list of [course, prereq] edges.",
        "can_finish",
        [
            {"args": [2, [[1, 0]]], "expected": True},
            {"args": [2, [[1, 0], [0, 1]]], "expected": False},
            {"args": [1, []], "expected": True},
        ],
        "graphs",
        "hard",
        "def can_finish(num_courses, prerequisites):\n"
        "    from collections import defaultdict, deque\n"
        "    graph = defaultdict(list)\n"
        "    indeg = [0] * num_courses\n"
        "    for a, b in prerequisites:\n"
        "        graph[b].append(a)\n"
        "        indeg[a] += 1\n"
        "    q = deque(i for i in range(num_courses) if indeg[i] == 0)\n"
        "    seen = 0\n"
        "    while q:\n"
        "        u = q.popleft()\n"
        "        seen += 1\n"
        "        for v in graph[u]:\n"
        "            indeg[v] -= 1\n"
        "            if indeg[v] == 0:\n"
        "                q.append(v)\n"
        "    return seen == num_courses",
    ),
    P(
        "h_trap_rain",
        "Write trap(height): units of rainwater trapped between bars (classic trapping rain water).",
        "trap",
        [
            {"args": [[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]], "expected": 6},
            {"args": [[4, 2, 0, 3, 2, 5]], "expected": 9},
            {"args": [[1, 2]], "expected": 0},
        ],
        "arrays",
        "hard",
        "def trap(height):\n"
        "    n = len(height)\n"
        "    if n < 3:\n"
        "        return 0\n"
        "    left, right = 0, n - 1\n"
        "    lmax = rmax = water = 0\n"
        "    while left < right:\n"
        "        if height[left] < height[right]:\n"
        "            lmax = max(lmax, height[left])\n"
        "            water += lmax - height[left]\n"
        "            left += 1\n"
        "        else:\n"
        "            rmax = max(rmax, height[right])\n"
        "            water += rmax - height[right]\n"
        "            right -= 1\n"
        "    return water",
    ),
    P(
        "h_longest_palindrome_sub",
        "Write longest_palindrome_subseq(s): length of longest palindromic subsequence.",
        "longest_palindrome_subseq",
        [
            {"args": ["bbbab"], "expected": 4},
            {"args": ["cbbd"], "expected": 2},
            {"args": ["a"], "expected": 1},
        ],
        "dp",
        "hard",
        "def longest_palindrome_subseq(s):\n"
        "    n = len(s)\n"
        "    dp = [[0] * n for _ in range(n)]\n"
        "    for i in range(n):\n"
        "        dp[i][i] = 1\n"
        "    for length in range(2, n + 1):\n"
        "        for i in range(n - length + 1):\n"
        "            j = i + length - 1\n"
        "            if s[i] == s[j]:\n"
        "                dp[i][j] = 2 + (dp[i + 1][j - 1] if length > 2 else 0)\n"
        "            else:\n"
        "                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])\n"
        "    return dp[0][n - 1]",
    ),
]

# Extra hard variants per topic so same_db_split has enough per topic
EXTRA_HARD = [
    (
        "h_fib_mod",
        "Write fib_mod(n, m): the n-th Fibonacci number (F0=0,F1=1) modulo m.",
        "fib_mod",
        [{"args": [10, 1000], "expected": 55}, {"args": [0, 7], "expected": 0}, {"args": [7, 10], "expected": 3}],
        "dp",
        "def fib_mod(n, m):\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 1):\n"
        "        a, b = b, (a + b) % m\n"
        "    return b % m",
    ),
    (
        "h_subset_sum",
        "Write subset_sum(nums, target): True if a subset sums to target.",
        "subset_sum",
        [
            {"args": [[3, 34, 4, 12, 5, 2], 9], "expected": True},
            {"args": [[3, 34, 4, 12, 5, 2], 30], "expected": False},
            {"args": [[], 0], "expected": True},
        ],
        "dp",
        "def subset_sum(nums, target):\n"
        "    possible = {0}\n"
        "    for x in nums:\n"
        "        possible |= {s + x for s in possible if s + x <= target}\n"
        "    return target in possible",
    ),
    (
        "h_decode_ways",
        "Write num_decodings(s): ways to decode a digit string with A=1..Z=26 mapping. "
        "Leading zeros invalid.",
        "num_decodings",
        [{"args": ["12"], "expected": 2}, {"args": ["226"], "expected": 3}, {"args": ["06"], "expected": 0}],
        "dp",
        "def num_decodings(s):\n"
        "    if not s or s[0] == '0':\n"
        "        return 0\n"
        "    n = len(s)\n"
        "    dp = [0] * (n + 1)\n"
        "    dp[0] = dp[1] = 1\n"
        "    for i in range(2, n + 1):\n"
        "        if s[i - 1] != '0':\n"
        "            dp[i] += dp[i - 1]\n"
        "        two = int(s[i - 2:i])\n"
        "        if 10 <= two <= 26:\n"
        "            dp[i] += dp[i - 2]\n"
        "    return dp[n]",
    ),
    (
        "h_rotate_right",
        "Write rotate_right(nums, k): return a new list rotated right by k steps.",
        "rotate_right",
        [
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [4, 5, 1, 2, 3]},
            {"args": [[1, 2], 3], "expected": [2, 1]},
            {"args": [[1], 0], "expected": [1]},
        ],
        "arrays",
        "def rotate_right(nums, k):\n"
        "    if not nums:\n"
        "        return []\n"
        "    k %= len(nums)\n"
        "    return nums[-k:] + nums[:-k] if k else list(nums)",
    ),
    (
        "h_spiral_order",
        "Write spiral_order(matrix): return elements of matrix in spiral order.",
        "spiral_order",
        [
            {"args": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], "expected": [1, 2, 3, 6, 9, 8, 7, 4, 5]},
            {"args": [[[1, 2], [3, 4]]], "expected": [1, 2, 4, 3]},
        ],
        "arrays",
        "def spiral_order(matrix):\n"
        "    if not matrix:\n"
        "        return []\n"
        "    out = []\n"
        "    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1\n"
        "    while top <= bottom and left <= right:\n"
        "        for j in range(left, right + 1):\n"
        "            out.append(matrix[top][j])\n"
        "        top += 1\n"
        "        for i in range(top, bottom + 1):\n"
        "            out.append(matrix[i][right])\n"
        "        right -= 1\n"
        "        if top <= bottom:\n"
        "            for j in range(right, left - 1, -1):\n"
        "                out.append(matrix[bottom][j])\n"
        "            bottom -= 1\n"
        "        if left <= right:\n"
        "            for i in range(bottom, top - 1, -1):\n"
        "                out.append(matrix[i][left])\n"
        "            left += 1\n"
        "    return out",
    ),
    (
        "h_valid_parens",
        "Write is_valid_parens(s): True if (), [], {} are correctly matched and nested.",
        "is_valid_parens",
        [
            {"args": ["()[]{}"], "expected": True},
            {"args": ["(]"], "expected": False},
            {"args": ["([)]"], "expected": False},
            {"args": ["{[]}"], "expected": True},
        ],
        "strings",
        "def is_valid_parens(s):\n"
        "    pair = {')': '(', ']': '[', '}': '{'}\n"
        "    stack = []\n"
        "    for c in s:\n"
        "        if c in '([{':\n"
        "            stack.append(c)\n"
        "        elif c in pair:\n"
        "            if not stack or stack[-1] != pair[c]:\n"
        "                return False\n"
        "            stack.pop()\n"
        "    return not stack",
    ),
    (
        "h_longest_unique",
        "Write length_of_longest_substring(s): length of longest substring without repeating chars.",
        "length_of_longest_substring",
        [
            {"args": ["abcabcbb"], "expected": 3},
            {"args": ["bbbbb"], "expected": 1},
            {"args": ["pwwkew"], "expected": 3},
            {"args": [""], "expected": 0},
        ],
        "strings",
        "def length_of_longest_substring(s):\n"
        "    last = {}\n"
        "    start = best = 0\n"
        "    for i, c in enumerate(s):\n"
        "        if c in last and last[c] >= start:\n"
        "            start = last[c] + 1\n"
        "        last[c] = i\n"
        "        best = max(best, i - start + 1)\n"
        "    return best",
    ),
    (
        "h_min_window_cover",
        "Write min_window_len(s, t): length of the minimum window in s covering all chars in t "
        "(including duplicates). Return 0 if impossible.",
        "min_window_len",
        [
            {"args": ["ADOBECODEBANC", "ABC"], "expected": 4},
            {"args": ["a", "a"], "expected": 1},
            {"args": ["a", "aa"], "expected": 0},
        ],
        "strings",
        "def min_window_len(s, t):\n"
        "    from collections import Counter\n"
        "    need = Counter(t)\n"
        "    missing = len(t)\n"
        "    best = float('inf')\n"
        "    start = 0\n"
        "    left = 0\n"
        "    for right, c in enumerate(s):\n"
        "        if need[c] > 0:\n"
        "            missing -= 1\n"
        "        need[c] -= 1\n"
        "        while missing == 0:\n"
        "            if right - left + 1 < best:\n"
        "                best = right - left + 1\n"
        "            need[s[left]] += 1\n"
        "            if need[s[left]] > 0:\n"
        "                missing += 1\n"
        "            left += 1\n"
        "    return 0 if best == float('inf') else best",
    ),
    (
        "h_oranges_rotting",
        "Write oranges_rotting(grid): minutes until all fresh oranges (1) rot from rotten (2). "
        "Empty=0. Return -1 if impossible.",
        "oranges_rotting",
        [
            {"args": [[[2, 1, 1], [1, 1, 0], [0, 1, 1]]], "expected": 4},
            {"args": [[[2, 1, 1], [0, 1, 1], [1, 0, 1]]], "expected": -1},
            {"args": [[[0, 2]]], "expected": 0},
        ],
        "graphs",
        "def oranges_rotting(grid):\n"
        "    from collections import deque\n"
        "    m, n = len(grid), len(grid[0])\n"
        "    q = deque()\n"
        "    fresh = 0\n"
        "    for i in range(m):\n"
        "        for j in range(n):\n"
        "            if grid[i][j] == 2:\n"
        "                q.append((i, j, 0))\n"
        "            elif grid[i][j] == 1:\n"
        "                fresh += 1\n"
        "    minutes = 0\n"
        "    while q:\n"
        "        i, j, d = q.popleft()\n"
        "        minutes = d\n"
        "        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n"
        "            ni, nj = i + di, j + dj\n"
        "            if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:\n"
        "                grid[ni][nj] = 2\n"
        "                fresh -= 1\n"
        "                q.append((ni, nj, d + 1))\n"
        "    return -1 if fresh else minutes",
    ),
    (
        "h_network_delay",
        "Write network_delay_time(n, times, k): time for signal from node k to reach all n nodes. "
        "times is list of [u, v, w] directed edges. Return -1 if impossible. Nodes are 1..n.",
        "network_delay_time",
        [
            {"args": [4, [[2, 1, 1], [2, 3, 1], [3, 4, 1]], 2], "expected": 2},
            {"args": [2, [[1, 2, 1]], 1], "expected": 1},
            {"args": [2, [[1, 2, 1]], 2], "expected": -1},
        ],
        "graphs",
        "def network_delay_time(n, times, k):\n"
        "    import heapq\n"
        "    from collections import defaultdict\n"
        "    g = defaultdict(list)\n"
        "    for u, v, w in times:\n"
        "        g[u].append((v, w))\n"
        "    dist = {k: 0}\n"
        "    pq = [(0, k)]\n"
        "    while pq:\n"
        "        d, u = heapq.heappop(pq)\n"
        "        if d > dist.get(u, 10**18):\n"
        "            continue\n"
        "        for v, w in g[u]:\n"
        "            nd = d + w\n"
        "            if nd < dist.get(v, 10**18):\n"
        "                dist[v] = nd\n"
        "                heapq.heappush(pq, (nd, v))\n"
        "    return max(dist.values()) if len(dist) == n else -1",
    ),
    (
        "h_partition_equal",
        "Write can_partition(nums): True if nums can be partitioned into two subsets with equal sum.",
        "can_partition",
        [
            {"args": [[1, 5, 11, 5]], "expected": True},
            {"args": [[1, 2, 3, 5]], "expected": False},
            {"args": [[1, 1]], "expected": True},
        ],
        "dp",
        "def can_partition(nums):\n"
        "    s = sum(nums)\n"
        "    if s % 2:\n"
        "        return False\n"
        "    target = s // 2\n"
        "    possible = {0}\n"
        "    for x in nums:\n"
        "        possible |= {p + x for p in possible if p + x <= target}\n"
        "    return target in possible",
    ),
    (
        "h_gas_station",
        "Write can_complete_circuit(gas, cost): starting gas station index to complete the circuit, "
        "or -1. Unique answer if exists.",
        "can_complete_circuit",
        [
            {"args": [[1, 2, 3, 4, 5], [3, 4, 5, 1, 2]], "expected": 3},
            {"args": [[2, 3, 4], [3, 4, 3]], "expected": -1},
        ],
        "greedy",
        "def can_complete_circuit(gas, cost):\n"
        "    if sum(gas) < sum(cost):\n"
        "        return -1\n"
        "    tank = start = 0\n"
        "    for i, (g, c) in enumerate(zip(gas, cost)):\n"
        "        tank += g - c\n"
        "        if tank < 0:\n"
        "            start = i + 1\n"
        "            tank = 0\n"
        "    return start",
    ),
]

for pid, q, fn, tests, topic, gold in EXTRA_HARD:
    PROBLEMS.append(P(pid, q, fn, tests, topic, "hard", gold))

# A few "extra" difficulty (harder held-out flavor)
PROBLEMS += [
    P(
        "x_median_sorted",
        "Write find_median_sorted(a, b): median of two sorted arrays a and b combined "
        "(float). Lengths may differ; total length >= 1.",
        "find_median_sorted",
        [
            {"args": [[1, 3], [2]], "expected": 2.0},
            {"args": [[1, 2], [3, 4]], "expected": 2.5},
            {"args": [[0, 0], [0, 0]], "expected": 0.0},
        ],
        "arrays",
        "extra",
        "def find_median_sorted(a, b):\n"
        "    nums = sorted(a + b)\n"
        "    n = len(nums)\n"
        "    mid = n // 2\n"
        "    if n % 2:\n"
        "        return float(nums[mid])\n"
        "    return (nums[mid - 1] + nums[mid]) / 2.0",
    ),
    P(
        "x_regex_match",
        "Write is_match(s, p): True if entire string s matches pattern p where '.' matches "
        "any char and '*' means zero-or-more of the preceding element.",
        "is_match",
        [
            {"args": ["aa", "a"], "expected": False},
            {"args": ["aa", "a*"], "expected": True},
            {"args": ["ab", ".*"], "expected": True},
            {"args": ["aab", "c*a*b"], "expected": True},
        ],
        "dp",
        "extra",
        "def is_match(s, p):\n"
        "    m, n = len(s), len(p)\n"
        "    dp = [[False] * (n + 1) for _ in range(m + 1)]\n"
        "    dp[0][0] = True\n"
        "    for j in range(2, n + 1):\n"
        "        if p[j - 1] == '*':\n"
        "            dp[0][j] = dp[0][j - 2]\n"
        "    for i in range(1, m + 1):\n"
        "        for j in range(1, n + 1):\n"
        "            if p[j - 1] == '*':\n"
        "                dp[i][j] = dp[i][j - 2]\n"
        "                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:\n"
        "                    dp[i][j] = dp[i][j] or dp[i - 1][j]\n"
        "            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:\n"
        "                dp[i][j] = dp[i - 1][j - 1]\n"
        "    return dp[m][n]",
    ),
    P(
        "x_max_path_tree",
        "Write max_path_sum_tree(n, edges, values): given a tree with n nodes (0..n-1), undirected "
        "edges as [u,v] pairs, and node values, return the maximum path sum (path may start/end "
        "anywhere). At least one node.",
        "max_path_sum_tree",
        [
            {"args": [3, [[0, 1], [1, 2]], [1, 2, 3]], "expected": 6},
            {"args": [1, [], [-3]], "expected": -3},
            {"args": [2, [[0, 1]], [2, -1]], "expected": 2},
        ],
        "graphs",
        "extra",
        "def max_path_sum_tree(n, edges, values):\n"
        "    from collections import defaultdict\n"
        "    g = defaultdict(list)\n"
        "    for u, v in edges:\n"
        "        g[u].append(v)\n"
        "        g[v].append(u)\n"
        "    best = [-10**18]\n"
        "    def dfs(u, parent):\n"
        "        top1 = top2 = 0\n"
        "        for v in g[u]:\n"
        "            if v == parent:\n"
        "                continue\n"
        "            child = max(0, dfs(v, u))\n"
        "            if child > top1:\n"
        "                top2, top1 = top1, child\n"
        "            elif child > top2:\n"
        "                top2 = child\n"
        "        best[0] = max(best[0], values[u] + top1 + top2)\n"
        "        return values[u] + top1\n"
        "    dfs(0, -1)\n"
        "    return best[0]",
    ),
]


def main() -> None:
    # Verify every gold solution passes its tests before writing.
    import sys

    sys.path.insert(0, str(REPO))
    from harness.sandbox import execution_accuracy

    failed = []
    for p in PROBLEMS:
        acc, valid, err = execution_accuracy(
            p["gold_solution"], p["function_name"], p["tests"]
        )
        if acc != 1.0:
            failed.append((p["id"], valid, err))

    if failed:
        print(f"Gold verification failed for {len(failed)} problems:")
        for pid, valid, err in failed[:20]:
            print(f"  {pid}: valid={valid} err={err}")
        raise SystemExit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(PROBLEMS, indent=2) + "\n", encoding="utf-8")
    easy = sum(1 for p in PROBLEMS if p["difficulty"] == "easy")
    hard = sum(1 for p in PROBLEMS if p["difficulty"] == "hard")
    extra = sum(1 for p in PROBLEMS if p["difficulty"] == "extra")
    topics = sorted({p["topic"] for p in PROBLEMS})
    print(f"Wrote {OUT} — {len(PROBLEMS)} problems (easy={easy}, hard={hard}, extra={extra})")
    print(f"Topics: {topics}")


if __name__ == "__main__":
    main()
