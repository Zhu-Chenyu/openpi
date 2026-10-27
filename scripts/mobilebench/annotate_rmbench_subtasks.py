"""Slow-memory phase sentences for any RMBench task from the collector's subtask annotations.

RMBench's script/collect_data.py writes language_annotation.json next to the demos: per episode a list
of [subtask sentence, frames] whose frame counts add up to the episode length. Consecutive repeats are
merged into steps; frame t of step k (1-based) gets

    "Step k: <current subtask> Before: <previous step's subtask>"     ("... Before: nothing." for k = 1)

The step number and the previous subtask are what the slow memory has to carry where the current
subtask alone is ambiguous (blocks_ranking_try presses the same button six times; place_block_mat
returns "the block on the green mat to the original mat" three times; the remembered order or mat is
only in the earlier steps).

Collected dirs (prep_rmbench_hdf5.py, with source_files) are labelled; other dirs of the same task (the
original GR00T demos carry no subtask annotation) get the same sentence table with every frame -1 =
unlabelled, so the trainer sees one table across its data dirs.

    python annotate_rmbench_subtasks.py --collected <prep dir> [...] --unlabelled <prep dir> [...]

Writes <dir>/phase_labels.npz: frame_phase [frames], texts, text_emb (frozen CLIP, annotate_phases.embed),
text_model, group = step number of each sentence, group_name "step" (logged as slow_step_acc).
"""

import argparse
import json
import os
import re

import numpy as np

import annotate_phases


def episode_texts(segments: list, n: int) -> list[str]:
    steps = []
    for text, k in segments:
        text = " ".join(str(text).split())
        if steps and steps[-1][0] == text:
            steps[-1][1] += int(k)
        else:
            steps.append([text, int(k)])
    if sum(k for _, k in steps) != n:
        raise ValueError(f"subtask frames {sum(k for _, k in steps)} != episode length {n}")
    out = []
    for i, (text, k) in enumerate(steps):
        before = steps[i - 1][0] if i else "nothing."
        out += [f"Step {i + 1}: {text} Before: {before}"] * k
    return out


def collected_texts(meta) -> list[str]:
    cache, out = {}, []
    for path, n in zip(meta["source_files"], meta["ep_len"]):
        path = str(path)
        root = os.path.dirname(os.path.dirname(path))
        if root not in cache:
            cache[root] = json.load(open(f"{root}/language_annotation.json"))
        ep = int(re.search(r"episode(\d+)\.hdf5$", path).group(1))
        out += episode_texts(cache[root][f"episode_{ep}"], int(n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", nargs="+", required=True)
    ap.add_argument("--unlabelled", nargs="*", default=[])
    a = ap.parse_args()
    per_dir = {d: collected_texts(np.load(f"{d}/meta.npz")) for d in a.collected}
    table = sorted({t for ts in per_dir.values() for t in ts}, key=lambda t: (int(t.split(":")[0][5:]), t))
    index = {t: i for i, t in enumerate(table)}
    emb = annotate_phases.embed(table)
    group = np.array([int(t.split(":")[0][5:]) for t in table])
    common = dict(texts=np.array(table), text_emb=emb, text_model=annotate_phases.TEXT_MODEL, group=group, group_name="step")
    for d, ts in per_dir.items():
        np.savez(f"{d}/phase_labels.npz", frame_phase=np.array([index[t] for t in ts]), **common)
    for d in a.unlabelled:
        n = int(np.load(f"{d}/meta.npz")["ep_len"].sum())
        np.savez(f"{d}/phase_labels.npz", frame_phase=np.full(n, -1), **common)
    counts = np.bincount([index[t] for ts in per_dir.values() for t in ts], minlength=len(table))
    print(f"{len(table)} sentences, steps 1..{group.max()}; most frequent:")
    for i in np.argsort(-counts)[:8]:
        print(f"  {counts[i]:7d}  {table[i]}")


if __name__ == "__main__":
    main()
