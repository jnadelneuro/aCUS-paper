"""Central path configuration for the aCUS-paper repo (Python side).

Usage in a script:
    from _config import DATA_ROOT, RI60_DIR, INTRINSIC_DIR, MODEL_DIR, RNASEQ_DIR
    df = pd.read_pickle(os.path.join(RI60_DIR, "behaviorData.pickle"))

This is the ONLY place data paths live. Reads config/config.yaml (copy it from
config/config.example.yaml). A copy of this file is placed in each src/ pipeline
folder so `import _config` resolves as a sibling.
"""
import os
import yaml


def _find_config():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        p = os.path.join(d, "config", "config.yaml")
        if os.path.exists(p):
            return p
        d = os.path.dirname(d)
    raise FileNotFoundError(
        "config/config.yaml not found — copy config/config.example.yaml to config/config.yaml"
    )


with open(_find_config()) as _f:
    _cfg = yaml.safe_load(_f)

DATA_ROOT = _cfg["paths"]["data_root"]
AVOIDANCE_ROOT = _cfg["paths"]["avoidance_root"]
FIG_OUT = _cfg["paths"]["fig_out"]

_sp = _cfg["subpaths"]
RI60_DIR = os.path.join(DATA_ROOT, _sp["ri60"])
INTRINSIC_DIR = os.path.join(DATA_ROOT, _sp["intrinsic"])
MODEL_DIR = os.path.join(DATA_ROOT, _sp["model"])
RNASEQ_DIR = os.path.join(DATA_ROOT, _sp["rnaseq"])

TOOLS = _cfg.get("tools", {})
