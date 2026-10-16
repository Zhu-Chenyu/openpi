"""Slow-memory phase annotations for RMBench put_back_block (MobileBench v1.11, sec. 3.6 / 6.2).

Every frame gets one short "phase - object - holding state - destination" sentence, built from
the recorded gripper events (every put_back_block demo is R-close, R-open, L-close, R-close,
R-open) and the mat the block was taken from:

    0 reach    [0, R1 close)            reaching for the block on its mat
    1 carry    [R1 close, R1 open)      carrying it to the center
    2 to_btn   [R1 open, L close)       block at the center, left arm going to the button
    3 press    [L close, press]         pressing the button (press = lowest left TCP)
    4 regrasp  (press, R2 close)        button pressed, right arm reaching the block again
    5 return   [R2 close, R2 open)      carrying the block back to its mat
    6 done     [R2 open, end]           block returned

The mat (left / right / front / back) is what the slow memory has to keep: from phase 2 on it
is no longer visible. Freshly collected demos name it in RMBench's language_annotation.json
("Move the block back to the <mat> mat."). The original GR00T-converted demos carry no such
label, so their mat is read from the pick geometry: the right TCP's planar offset from the
center drop-off to the pick, classified by the per-mat mean offsets of the collected demos
(which are checked against their own language labels first).

Texts are embedded once with a frozen CLIP text encoder (openai/clip-vit-base-patch32,
512-d, L2-normalised); the trainer only reads the stored embeddings. All data dirs share the
same 28-sentence table (7 phases x 4 mats, in that order).

    python annotate_phases.py <collected prep dir> [<other prep dir> ...]
        (the first dir must come from prep_rmbench_hdf5.py: it calibrates the mat offsets)
Writes <dir>/phase_labels.npz: frame_phase [total frames] int (-1 = unlabelled episode),
ep_mat [episodes] int, texts [28], text_emb [28, 512], phases, mats.
"""

import argparse
import json
import os
import re

import numpy as np

MATS = ("left", "right", "front", "back")
PHASES = ("reach", "carry", "to_button", "press", "regrasp", "return", "done")
TEMPLATES = (
    "Reaching for the red block on the {m} mat; nothing held yet.",
    "Holding the red block taken from the {m} mat; carrying it to the center.",
    "Red block left at the center, taken from the {m} mat; left arm moving to the button.",
    "Pressing the button; the red block at the center belongs on the {m} mat.",
    "Button pressed; right arm reaching for the center block to return it to the {m} mat.",
    "Holding the red block; carrying it back to the {m} mat; placement not done yet.",
    "Red block returned to the {m} mat; task complete.",
)
PATTERN = [(1, 1), (1, 0), (0, 1), (1, 1), (1, 0)]  # (arm, closed): R-close R-open L-close R-close R-open
TEXT_MODEL = "openai/clip-vit-base-patch32"


def texts() -> list[str]:
    return [t.format(m=m) for t in TEMPLATES for m in MATS]


def episode_events(meta) -> dict[int, list[tuple[int, int, int]]]:
    ev = {}
    for e, f, a, c in meta["events"]:
        ev.setdefault(int(e), []).append((int(f), int(a), int(c)))
    return {e: sorted(v) for e, v in ev.items()}


def boundaries(ev, tcp_left_z: np.ndarray) -> list[int] | None:
    """[r1c, r1o, lc, press, r2c, r2o] frames, or None if the episode deviates from the pattern."""
    if [(a, c) for _, a, c in ev] != PATTERN:
        return None
    r1c, r1o, lc, r2c, r2o = (f for f, _, _ in ev)
    press = lc + int(np.argmin(tcp_left_z[lc:r2c]))
    return [r1c, r1o, lc, press, r2c, r2o]


def pick_offset(tcp_pos: np.ndarray, s0: int, b: list[int]) -> np.ndarray:
    """Planar right-TCP offset pick - center drop-off (robot footprint frame)."""
    return tcp_pos[s0 + b[0], 1, :2] - tcp_pos[s0 + b[1], 1, :2]


def collected_mats(meta) -> np.ndarray:
    """Mat index per episode of a prep_rmbench_hdf5.py dir, from RMBench's language annotations."""
    cache, out = {}, []
    for path in meta["source_files"]:
        path = str(path)
        root = os.path.dirname(os.path.dirname(path))
        if root not in cache:
            cache[root] = json.load(open(f"{root}/language_annotation.json"))
        ep = int(re.search(r"episode(\d+)\.hdf5$", path).group(1))
        names = {m for text, _ in cache[root][f"episode_{ep}"] for m in re.findall(r"back to the (\w+) mat", text)}
        assert len(names) == 1, (path, names)
        out.append(MATS.index(names.pop()))
    return np.array(out)


def frame_phases(n: int, b: list[int]) -> np.ndarray:
    r1c, r1o, lc, press, r2c, r2o = b
    edges = [r1c, r1o, lc, press + 1, r2c, r2o]  # phase p covers [edges[p-1], edges[p])
    return np.searchsorted(np.array(edges), np.arange(n), side="right")


def embed(sentences: list[str]) -> np.ndarray:
    import torch
    from transformers import CLIPTextModelWithProjection
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(TEXT_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(TEXT_MODEL).eval()
    with torch.no_grad():
        e = model(**tok(sentences, padding=True, return_tensors="pt")).text_embeds
    return torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    metas = [dict(np.load(f"{d}/meta.npz")) for d in a.dirs]
    assert "source_files" in metas[0], "the first dir must be a collected (prep_rmbench_hdf5.py) dir"

    # Calibrate per-mat offsets on the collected demos, check them against their own labels.
    m0, ev0 = metas[0], episode_events(metas[0])
    mats0 = collected_mats(m0)
    offs = []
    for i, s0 in enumerate(m0["ep_start"]):
        b = boundaries(ev0[i], m0["tcp_pos"][s0 : s0 + m0["ep_len"][i], 0, 2])
        offs.append(pick_offset(m0["tcp_pos"], int(s0), b) if b else np.full(2, np.nan))
    offs = np.array(offs)
    ok = ~np.isnan(offs[:, 0])
    centers = np.stack([offs[ok & (mats0 == k)].mean(0) for k in range(len(MATS))])
    print("mean pick offset per mat (m):", {m: np.round(c, 3).tolist() for m, c in zip(MATS, centers)})

    def classify(off):
        d = np.linalg.norm(centers - off, axis=-1)
        return int(np.argmin(d)), float(np.sort(d)[1] - np.sort(d)[0])

    acc = np.mean([classify(o)[0] == m for o, m, k in zip(offs, mats0, ok) if k])
    print(f"geometry vs language label on {a.dirs[0]}: {100 * acc:.1f}% agree")
    assert acc == 1.0, "pick-offset classification disagrees with the language labels"

    sentences = texts()
    emb = embed(sentences)
    for d, meta in zip(a.dirs, metas):
        ev = episode_events(meta)
        n_eps = len(meta["ep_index"])
        frame_phase = np.full(len(meta["state"]), -1, np.int64)
        ep_mat = np.full(n_eps, -1, np.int64)
        margins, skipped = [], 0
        lang = collected_mats(meta) if "source_files" in meta else None
        for i in range(n_eps):
            s0, n = int(meta["ep_start"][i]), int(meta["ep_len"][i])
            b = boundaries(ev.get(i, []), meta["tcp_pos"][s0 : s0 + n, 0, 2])
            if b is None:
                skipped += 1
                continue
            mat, margin = classify(pick_offset(meta["tcp_pos"], s0, b))
            margins.append(margin)
            if lang is not None:
                assert mat == lang[i], (d, i, MATS[mat], MATS[lang[i]])
            ep_mat[i] = mat
            frame_phase[s0 : s0 + n] = frame_phases(n, b) * len(MATS) + mat
        np.savez(
            f"{d}/phase_labels.npz", frame_phase=frame_phase, ep_mat=ep_mat, texts=np.array(sentences),
            text_emb=emb, phases=np.array(PHASES), mats=np.array(MATS), text_model=np.array(TEXT_MODEL),
        )  # fmt: skip
        counts = np.bincount(ep_mat[ep_mat >= 0], minlength=len(MATS))
        print(f"{d}: {n_eps - skipped}/{n_eps} episodes labelled, mats {dict(zip(MATS, counts.tolist()))}, "
              f"smallest classification margin {min(margins):.3f} m"
              f"{' (language-checked)' if lang is not None else ''}")  # fmt: skip
        ph = frame_phase[frame_phase >= 0] // len(MATS)
        print("  frames per phase:", dict(zip(PHASES, np.bincount(ph, minlength=len(PHASES)).tolist())))


if __name__ == "__main__":
    main()
