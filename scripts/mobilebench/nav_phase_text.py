# Vendored unchanged from Memory-Augmented-VLA scripts/magicsim/nav_phase_text.py @ 670c753 (prep_magicsim_nav.py uses it).
"""Phase-text labels for the SLOW memory (design doc sec. 7) of MagicSim go2 navigation episodes.

Generated from logged simulator state, not written by hand. For every policy update
(every `stride` frames) the episode's task facts are computed causally -- only from
the robot's motion so far. Per update:

  update, frame   update index and simulator frame (50 fps) of its image. The slow group is
                  written when update % slow_write_every == 0 (file header); the phase-text
                  loss uses only those updates, the rest serve the phase-class control and
                  other K values without relabelling.
  facts           phase, target, destination, holding (None = nothing; kept for format
                  parity with manipulation), obstacle, detour side, completed_events (causal,
                  ordered), distance. Phase class = header["phases"].index(phase).
  positives       3 semantically equivalent phrasings  (P_t: never negatives of each other)
  hard_negatives  same template with ONE task fact changed (phase, detour side, target,
                  holding, destination, completed events) -- sec. 7: "change the fact, not
                  the wording"

The text is a TRAINING TARGET for the slow-memory readout only; the policy's prompt
stays the original instruction.

Phases: 0 start, 1 approach (table ahead), 2 passing the table, 3 table cleared,
4 final approach, 5 arrived. Distances for the final approach / arrival are measured to the
NavTo stop point (the planner goal in front of the target), so they do not depend on the
target's size. The obstacle is always the coffee table; other scene furniture only enters
the hard negatives (wrong target / destination). The detour side becomes known only once the robot has
committed to it (lateral offset from the table > COMMIT_M on the final side), and is
then REMEMBERED in every later text -- the fact the slow group must carry. "Committed"
means the robot is laterally outside the table's half-width (+ COMMIT_MARGIN_M) on that
side, or already beside the table; a robot that starts that far out has its side known
from t=0, which is a geometric fact, not a leak.

Usage: python nav_phase_text.py <episode_dir> [out.json] [--target fridge] [--stride 50] [--k 4]
"""

import argparse
import json
import math

import numpy as np

PHASES = ("start", "approach", "passing", "cleared", "final_approach", "arrived")
COMMIT_MARGIN_M = 0.20  # a side is committed once the robot is this far outside the table's half-width
PASS_MARGIN_M = 0.30  # along-track margin around the table counted as "passing"
FINAL_M = 0.7  # distance to the NavTo stop point that starts the final approach
ARRIVE_M = 0.35  # the last update is up to 1 s before the end, so "arrived" allows some slack
HOLDING_NEGATIVE = "a small box"  # navigation carries nothing; the holding negative claims an object


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def scene_to_world(rel, base_pos0, base_quat0):
    """scene.json object positions are in the robot's frame at frame 0 (its ACTUAL pose,
    yaw included; scene["robots"][0]["base_pose"] is stale). Verified against the cube's
    logged world pose to ~2 cm."""
    w, x, y, z = base_quat0
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    rel = np.asarray(rel[:2], float)
    return np.asarray(base_pos0[:2], float) + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])


def load_episode(ep_dir: str, target: str = "fridge") -> dict:
    info = json.load(open(f"{ep_dir}/env/info/env_0_merged.json"))
    col = json.load(open(f"{ep_dir}/collect/collect_merged.json"))
    scene = json.load(open(f"{ep_dir}/language/scene.json"))
    frames = sorted(info)
    pos, yaw = [], []
    for f in frames:
        rs = info[f]["obs"]["policy_obs"]["robot_state"][0]["Robot1"]
        pos.append(rs["base_pos"][:2])
        yaw.append(yaw_of(rs["base_quat"]))
    rs0 = info[frames[0]]["obs"]["policy_obs"]["robot_state"][0]["Robot1"]
    obj = {o["cat_name"]: o for o in scene["objects"]}
    goal = np.array(col[frames[0]]["global_planner_info"]["command"][2][:2])  # NavTo stop point
    table = obj["coffee_table"]
    table_xy = scene_to_world(table["pos"], rs0["base_pos"], rs0["base_quat"])
    return {
        "pos": np.array(pos),
        "yaw": np.array(yaw),
        "goal": goal,
        "target_xy": scene_to_world(obj[target]["pos"], rs0["base_pos"], rs0["base_quat"]),
        "target_noun": obj[target]["noun"],
        # other furniture, for wrong-target / wrong-destination negatives
        "distractor_nouns": sorted({o["noun"] for k, o in obj.items() if k not in (target, "coffee_table")}),
        "table": table_xy,
        "table_extent": np.array(table["bbox"]["extent"][:2]),
        "fps": 50,
    }


def side_word(s: int) -> str:
    return "left" if s > 0 else "right"


def _core(f: dict) -> list[str]:
    """3 equivalent phrasings of the phase / target / obstacle / detour-side part."""
    tgt, obs = f["target"], f["obstacle"]
    side = f["detour_side"]
    passed = f"passed the {obs} on its {side}" if side else None
    p = f["phase"]
    if p == "start":
        return [
            f"Starting out toward the {tgt}; the {obs} is between us.",
            f"Just began heading for the {tgt}, with the {obs} in the way.",
            f"Setting off to the {tgt}; a {obs} blocks the direct path.",
        ]
    if p == "approach":
        if side:
            return [
                f"Approaching the {obs} on the way to the {tgt}; going around it on the {side}.",
                f"Nearing the {obs}; detouring to its {side} before continuing to the {tgt}.",
                f"Heading for the {tgt}, about to pass the {obs} on the {side}.",
            ]
        return [
            f"Approaching the {obs} on the way to the {tgt}; detour side not chosen yet.",
            f"Nearing the {obs} that stands between me and the {tgt}.",
            f"Heading for the {tgt} with the {obs} still ahead.",
        ]
    if p == "passing":
        return [
            f"Passing the {obs} on its {side}, heading to the {tgt}.",
            f"Going around the {obs} on the {side} side toward the {tgt}.",
            f"Beside the {obs} on its {side}; the {tgt} is beyond it.",
        ]
    if p == "cleared":
        return [
            f"Cleared the {obs} on its {side}; now heading straight to the {tgt}.",
            f"The {obs} is behind me ({passed}); continuing to the {tgt}.",
            f"After going around the {obs} on the {side}, walking toward the {tgt}.",
        ]
    if p == "final_approach":
        return [
            f"Close to the {tgt}, slowing for the final approach; {passed}.",
            f"Final approach to the {tgt}; the {obs} was passed on the {side}.",
            f"Almost at the {tgt}; earlier went around the {obs} on the {side}.",
        ]
    return [
        f"Arrived at the {tgt}; the {obs} was passed on the {side}.",
        f"Reached the {tgt} after going around the {obs} on the {side}.",
        f"Stopped at the {tgt}; the route went around the {obs}'s {side} side.",
    ]


EVENTS = ("set_off", "reached_obstacle", "passed_obstacle", "arrived")


def event_text(e: str, f: dict) -> str:
    return {
        "set_off": "set off",
        "reached_obstacle": f"reached the {f['obstacle']}",
        "passed_obstacle": f"went around the {f['obstacle']} on the {f['detour_side']}",
        "arrived": f"arrived at the {f['target']}",
    }[e]


def render(f: dict) -> list[str]:
    """3 equivalent phrasings of the full fact set: phase, target, obstacle / detour side,
    holding state, destination and completed events (sec. 7: phase-object-destination)."""
    hold = f["holding"] or "nothing"
    dest = f["destination"]
    done = ", ".join(event_text(e, f) for e in f["completed_events"]) or "nothing yet"
    tails = [
        f" Carrying {hold}; destination: {dest}; done so far: {done}.",
        f" Holding {hold}, headed for {dest}; completed: {done}.",
        f" Hands: {hold}. Going to: {dest}. Already done: {done}.",
    ]
    return [c + t for c, t in zip(_core(f), tails, strict=True)]


def completed_events(phase: str, ui: int) -> list[str]:
    """Events finished BEFORE this update, in order (causal)."""
    i = PHASES.index(phase)
    ev = []
    if ui > 0:
        ev.append("set_off")
    if i >= PHASES.index("passing"):
        ev.append("reached_obstacle")
    if i >= PHASES.index("cleared"):
        ev.append("passed_obstacle")
    if phase == "arrived":
        ev.append("arrived")
    return ev


def hard_negatives(f: dict, distractors: list[str]) -> list[dict]:
    """Same template family, exactly one task fact changed (a wrong target also moves the destination
    with it, since the destination is defined by the target)."""
    out = []

    def add(changed, **kw):
        g = {**f, **kw}
        out.append({"changed": changed, "text": render(g)[0]})

    other_side = "right" if f["detour_side"] == "left" else "left"

    if f["detour_side"]:
        add("detour_side", detour_side="right" if f["detour_side"] == "left" else "left")
    i = PHASES.index(f["phase"])
    for j in (i - 1, i + 1):
        if 0 <= j < len(PHASES):
            g = PHASES[j]
            needs_side = g in ("passing", "cleared", "final_approach", "arrived")
            add("phase", phase=g, detour_side=f["detour_side"] or ("left" if needs_side else None),
                completed_events=completed_events(g, 1))  # fmt: skip
    for d in distractors:
        add("target", target=d, destination=f"in front of the {d}")
    add("holding", holding=HOLDING_NEGATIVE)
    add("destination", destination=f"in front of the {distractors[0] if distractors else f['obstacle']}")
    ev = f["completed_events"]
    future = [e for e in EVENTS if e not in ev and e != "arrived"]
    if future:  # claims an event that has not happened yet
        add("completed_events", completed_events=[*ev, future[0]], detour_side=f["detour_side"] or other_side)
    elif ev:  # forgets the latest event
        add("completed_events", completed_events=ev[:-1])
    return out


def annotate(ep: dict, stride: int, k: int) -> list[dict]:
    pos, goal, table = ep["pos"], ep["goal"], ep["table"]
    start = pos[0]
    u = (goal - start) / np.linalg.norm(goal - start)  # along-track unit vector
    n = np.array([-u[1], u[0]])  # left normal
    half_along = abs(ep["table_extent"] @ np.abs(u)) / 2  # table half-depth along the track
    half_cross = abs(ep["table_extent"] @ np.abs(n)) / 2  # table half-width across the track
    along_t = (pos - table) @ u
    lateral_t = (pos - table) @ n  # >0: robot is LEFT of the table's centre line
    dist = np.linalg.norm(pos - goal, axis=1)  # to the NavTo stop point (phase thresholds)
    dist_obj = np.linalg.norm(pos - ep["target_xy"], axis=1)  # to the target itself (reported fact)
    tgt = ep["target_noun"]

    # The side the robot finally passes on (at closest along-track approach to the table).
    final_side = int(np.sign(lateral_t[np.argmin(np.abs(along_t))])) or 1
    committed = np.zeros(len(pos), bool)
    for t in range(len(pos)):
        committed[t] = committed[t - 1] if t else False
        if np.sign(lateral_t[t]) == final_side and abs(lateral_t[t]) > half_cross + COMMIT_MARGIN_M and along_t[t] < 0:
            committed[t] = True
        if along_t[t] >= -half_along - PASS_MARGIN_M:  # beside / past the table: side is observed
            committed[t] = True

    labels = []
    updates = list(range(0, len(pos), stride))
    for ui, t in enumerate(updates):
        last = ui == len(updates) - 1
        if ui == 0:
            phase = "start"
        elif last and dist[t] < ARRIVE_M:
            phase = "arrived"
        elif dist[t] < FINAL_M and along_t[t] > half_along:
            phase = "final_approach"
        elif along_t[t] < -half_along - PASS_MARGIN_M:
            phase = "approach"
        elif along_t[t] <= half_along + PASS_MARGIN_M:
            phase = "passing"
        else:
            phase = "cleared"
        side = side_word(final_side) if committed[t] else None
        facts = {
            "phase": phase,
            "target": tgt,
            "destination": f"in front of the {tgt}",
            "holding": None,  # navigation carries nothing; kept so manipulation tasks share the format
            "obstacle": "coffee table",
            "detour_side": side,
            "completed_events": completed_events(phase, ui),
            "dist_to_target_m": round(float(dist_obj[t]), 2),
        }
        labels.append(
            {
                "update": ui,  # slow group written when update % slow_write_every == 0
                "frame": int(t),  # simulator frame (50 fps) -> image in the onboard video
                "facts": facts,  # phase class for the control experiment = phases.index(phase)
                "positives": render(facts),
                "hard_negatives": hard_negatives(facts, ep["distractor_nouns"]),
            }
        )
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--target", default="fridge", help="scene category of the NavTo target")
    ap.add_argument("--stride", type=int, default=50, help="frames per policy update (50 = 1 s at 50 fps)")
    ap.add_argument("--k", type=int, default=4, help="slow group written every k-th update")
    args = ap.parse_args()
    ep = load_episode(args.episode_dir, args.target)
    out = {
        "instruction": f"Navigate to the {ep['target_noun']}.",
        "stride_frames": args.stride,
        "slow_write_every": args.k,
        "phases": list(PHASES),
        "updates": annotate(ep, args.stride, args.k),
    }
    s = json.dumps(out, indent=1)
    if args.out:
        open(args.out, "w").write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
