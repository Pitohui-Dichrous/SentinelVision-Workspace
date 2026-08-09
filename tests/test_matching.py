from __future__ import annotations

import unittest
import random

from safety_pipeline.matching import (
    maximum_cardinality_weight_matching,
    maximum_total_weight_matching,
)


class BipartiteMatchingTests(unittest.TestCase):
    def test_cardinality_wins_over_single_highest_weight_edge(self):
        # A greedy maximum-edge choice A-X=1.0 leaves B unmatched.  The correct
        # maximum-cardinality solution is A-Y plus B-X.
        result = maximum_cardinality_weight_matching(
            ("A", "B"),
            ("X", "Y"),
            {
                ("A", "X"): 1.000,
                ("A", "Y"): 0.724,
                ("B", "X"): 0.818,
            },
        )
        self.assertEqual(result, {"A": "Y", "B": "X"})

    def test_maximum_weight_breaks_equal_cardinality_choice(self):
        result = maximum_cardinality_weight_matching(
            (1, 2),
            (10, 20),
            {
                (1, 10): 0.9,
                (1, 20): 0.8,
                (2, 10): 0.7,
                (2, 20): 0.1,
            },
        )
        self.assertEqual(result, {1: 20, 2: 10})

    def test_optional_unmatched_prefers_one_strong_pair_over_two_weak_pairs(self):
        result = maximum_total_weight_matching(
            ("A", "B"),
            ("X", "Y"),
            {
                ("A", "X"): 1.000,
                ("A", "Y"): 0.316,
                ("B", "X"): 0.307,
            },
        )
        self.assertEqual(result, {"A": "X"})

    def test_optional_unmatched_keeps_two_pairs_when_their_total_is_better(self):
        result = maximum_total_weight_matching(
            ("A", "B"),
            ("X", "Y"),
            {
                ("A", "X"): 0.900,
                ("A", "Y"): 0.600,
                ("B", "X"): 0.600,
            },
        )
        self.assertEqual(result, {"A": "Y", "B": "X"})

    def test_small_random_graphs_match_exhaustive_optimum(self):
        generator = random.Random(20260809)

        def exhaustive(left, right, weights):
            best_count = -1
            best_weight = float("-inf")
            def visit(index, used, total, count):
                nonlocal best_count, best_weight
                if index == len(left):
                    if count > best_count or (count == best_count and total > best_weight):
                        best_count, best_weight = count, total
                    return
                visit(index + 1, used, total, count)
                for right_node in right:
                    pair = (left[index], right_node)
                    if right_node in used or pair not in weights:
                        continue
                    visit(index + 1, used | {right_node}, total + weights[pair], count + 1)
            visit(0, set(), 0.0, 0)
            return best_count, best_weight

        for _ in range(120):
            left = tuple(range(generator.randint(1, 4)))
            right = tuple(range(generator.randint(1, 4)))
            weights = {
                (left_node, right_node): generator.random()
                for left_node in left
                for right_node in right
                if generator.random() < 0.7
            }
            result = maximum_cardinality_weight_matching(left, right, weights)
            expected_count, expected_weight = exhaustive(left, right, weights)
            actual_weight = sum(weights[pair] for pair in result.items())
            self.assertEqual(len(result), expected_count)
            self.assertAlmostEqual(actual_weight, expected_weight, places=10)

    def test_optional_unmatched_random_graphs_match_exhaustive_total_weight(self):
        generator = random.Random(20260810)

        def exhaustive(left, right, weights):
            best_weight = 0.0
            def visit(index, used, total):
                nonlocal best_weight
                if index == len(left):
                    best_weight = max(best_weight, total)
                    return
                visit(index + 1, used, total)
                for right_node in right:
                    pair = (left[index], right_node)
                    if right_node in used or pair not in weights:
                        continue
                    visit(index + 1, used | {right_node}, total + weights[pair])
            visit(0, set(), 0.0)
            return best_weight

        for _ in range(120):
            left = tuple(range(generator.randint(1, 4)))
            right = tuple(range(generator.randint(1, 4)))
            weights = {
                (left_node, right_node): generator.uniform(-1.0, 1.0)
                for left_node in left
                for right_node in right
                if generator.random() < 0.7
            }
            result = maximum_total_weight_matching(left, right, weights)
            expected_weight = exhaustive(left, right, weights)
            actual_weight = sum(weights[pair] for pair in result.items())
            self.assertAlmostEqual(actual_weight, expected_weight, places=10)


if __name__ == "__main__":
    unittest.main()
