import json
from argparse import ArgumentParser, Namespace, FileType
import numpy as np
from sklearn.metrics import accuracy_score


def compute_score(scores: str):
    scores = np.array([float(i) for i in scores.strip().split()])
    return np.sum(scores)


def main(args: Namespace):
    contra_data = json.load(args.json)
    score_data = open(args.score, "r").readlines()
    score_data = [compute_score(i) for i in score_data]
    trail_num = [len(i["errors"]) + 1 for i in contra_data]
    assert sum(trail_num) == len(score_data), f"{sum(trail_num)} != {len(score_data)}"

    y_label = np.zeros(len(trail_num))
    y_pred = []
    pointer = 0
    for trail in trail_num:
        scores = np.array(score_data[pointer: pointer + trail])
        y_pred.append(np.argmax(scores))
        pointer += trail
    assert len(y_pred) == len(y_label)

    acc = accuracy_score(y_label, y_pred)
    print(f"Acc: {acc:2f}")
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json",
        type=FileType("r"),
        required=True,
        help="path to contra pro json file",
    )
    parser.add_argument(
        "--score",
        type=str,
        required=True,
        help="path to score file",
    )
    args = parser.parse_args()
    main(args)
