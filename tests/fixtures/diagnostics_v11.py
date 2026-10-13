"""Tiny original wire fixtures and mutations from issue #30's literal oracles.

No SDK/model execution or implementation-derived numerical expectations.
"""

from copy import deepcopy


def f32(hex_data, length=2):
    return {"dtype": "f32", "shape": [length], "data": bytes.fromhex(hex_data)}


def extend_attempt(base, *, visual=True):
    attempt = deepcopy(base)
    z = f32("0000803f00000040")  # [1, 2]
    h = f32("000000000000803e000080be", 3)  # [0, .25, -.25]
    attempt["initial_representation"] = {"z": deepcopy(z), "h": deepcopy(h)}
    step = attempt["steps"][0]
    step["policy_mode"] = "random"
    step["representation"] = {"z": f32("000000400000803f"), "h": deepcopy(h)}  # [2, 1]
    step["prediction"] = {
        "target": "post_action_last_frame",
        "latent": f32("0000c03f0000803f"),  # [1.5, 1], recorded successor
        "dynamics": {"z": z, "h": h, "delta_z": f32("0000003f000080bf"),
                     "h_next": f32("0000803e0000003f000080be", 3)},
        "errors": {"latent_mse": 0.125, "no_change_mse": 1.0, "future_metric": 0.3},
        "future_prediction": {"raw": b"\x01", "available": None},
    }
    if visual:
        step["observation"]["frames"] = {"dtype": "u8", "shape": [2, 2, 2],
                                         "data": bytes.fromhex("0001020300040500")}
        step["prediction"].update({
            "frames": {"dtype": "u8", "shape": [2, 2], "data": bytes.fromhex("00040200")},
            "visual_delta": {
                "reference_position": 0,
                "reference_frame": {"dtype": "u8", "shape": [2, 2], "data": bytes.fromhex("00010203")},
                "delta": {"dtype": "u8", "shape": [2, 2], "data": bytes.fromhex("10041000")},
            },
        })
    step["learning"] = {"modules": {
        "world_model": {"updates": 2, "loss": 0.125, "losses": {"latent_mse": 0.125, "sigreg": -0.25}},
        "decoder": {"updates": 1, "losses": {"color_cross_entropy": 2.0}},
        "future_module": {"future": None},
    }}
    return attempt


def invalid_cases():
    """Each mutation leaves other constraints valid to reach the owning guard."""
    s = ("steps", 0)
    p = (*s, "prediction")
    d = (*p, "dynamics")
    v = (*p, "visual_delta")
    m = (*s, "learning", "modules")
    return [
        ("initial-null", ("initial_representation",), None, "map"),
        ("state-dtype", (*s, "representation", "z"), {"dtype": "u8", "shape": [2], "data": b"\x01\x02"}, "f32"),
        ("state-rank", (*s, "representation", "h", "shape"), [1, 3], "rank"),
        ("state-zero-length", (*s, "representation", "z"), {"dtype": "f32", "shape": [0], "data": b""}, "nonpositive"),
        ("state-byte-length", (*d, "z", "data"), b"\x00", "BIN length"),
        ("state-nonfinite", (*d, "z", "data"), bytes.fromhex("0000c07f00000040"), "nonfinite"),
        ("latent-dimension", (*p, "latent"), f32("0000c03f0000803f00000000", 3), "latent dimensions"),
        ("residual-dimension", (*d, "delta_z"), f32("0000003f", 1), "latent dimensions"),
        ("memory-dimension", (*d, "h_next"), f32("0000003f000080bf"), "memory dimensions"),
        ("conditional-latent-dtype", (*p, "latent"), {"dtype": "f16", "shape": [2], "data": bytes.fromhex("003e003c")}, "f32"),
        ("dynamics-null", (*p, "dynamics"), None, "map"),
        ("latent-error-pair", (*s, "representation", "z"), f32("00000040", 1), "matched"),
        ("latent-error-negative", (*p, "errors", "latent_mse"), -0.1, "number"),
        ("policy-mode", (*s, "policy_mode"), "guess", "expected"),
        ("visual-null", (*p, "visual_delta"), None, "map"),
        ("future-reference", (*v, "reference_position"), 1, "integer"),
        ("float-reference", (*v, "reference_position"), 0.0, "integer"),
        ("keep-code", (*v, "delta", "data"), bytes.fromhex("11041000"), "palette/KEEP"),
        ("reference-color", (*v, "reference_frame", "data"), bytes.fromhex("10010203"), "palette/KEEP"),
        ("reference-mismatch", (*v, "reference_frame", "data"), bytes.fromhex("01010203"), "reference"),
        ("visual-geometry", (*v, "delta", "shape"), [1, 4], "geometry"),
        ("visual-dtype", (*v, "delta"), {"dtype": "f32", "shape": [2, 2], "data": bytes(16)}, "palette/KEEP"),
        ("decoded-board", (*p, "frames", "data"), bytes.fromhex("00040300"), "KEEP/replacement"),
        ("decoded-color", (*p, "frames", "data"), bytes.fromhex("00041000"), "colors"),
        ("modules-null", (*s, "learning", "modules"), None, "map"),
        ("modules-empty", (*s, "learning", "modules"), {}, "nonempty"),
        ("module-empty", (*m, "world_model"), {}, "reported data"),
        ("module-update-negative", (*m, "world_model", "updates"), -1, "integer"),
        ("module-losses-empty", (*m, "world_model", "losses"), {}, "nonempty"),
        ("module-losses-nonfinite", (*m, "world_model", "losses", "sigreg"), float("inf"), "nonfinite|finite"),
        ("world-model-reset", (*m, "world_model", "reset_reason"), "unexpected", "actor/critic"),
        ("actor-reset-empty", (*m, "actor"), {"reset_reason": ""}, "nonempty"),
    ]


def replace(attempt, path, value):
    for key in path[:-1]:
        attempt = attempt[key]
    attempt[path[-1]] = deepcopy(value)
