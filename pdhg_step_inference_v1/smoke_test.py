from core import choose_depth, paired_budget_rows


def main() -> None:
    assert choose_depth({1: 0.2, 4: 0.7, 8: 1.4}, 0.7) == 4
    assert choose_depth({1: 0.2}, 0.1) is None
    rows = paired_budget_rows({1: 0.2, 4: 0.7}, {1: 0.3, 4: 0.8}, [0.5, 1.0])
    assert rows[0]["learned_max_k"] == 1 and rows[1]["fixed_max_k"] == 4
    print("[pdhg-step-inference-smoke] PASS")


if __name__ == "__main__":
    main()
