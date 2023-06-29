# import rand
import numpy as np

from lib.modules.core import rand
from lib.modules.core.sample_range import (
    BOT_MUL,
    DEPTH,
    GROUP_W,
    NUM_STAGES,
    W0,
    WA,
    WM,
)


def regnet_sampler():
    """Sampler for main RegNet parameters."""
    d = rand.uniform(*DEPTH, *[1])
    w0 = rand.log_uniform(
        *W0,
        8,
    )
    wa = rand.log_uniform(
        *WA,
        0.1,
    )
    wm = rand.log_uniform(
        *WM,
        0.001,
    )
    gw = rand.power2_or_log_uniform(
        *GROUP_W,
        8,
    )
    bm = rand.power2_uniform(
        *BOT_MUL,
        1 / 128,
    )
    params = ["DEPTH", d, "W0", w0, "WA", wa, "WM", wm, "GROUP_W", gw, "BOT_MUL", bm]
    params = ["REGNET." + p if i % 2 == 0 else p for i, p in enumerate(params)]
    return dict(zip(params[::2], params[1::2]))


# print(regnet_sampler())
def check_regnet_constraints(param):
    """Checks RegNet specific constraints."""

    wa, w0, wm, d = (
        param["REGNET.WA"],
        param["REGNET.W0"],
        param["REGNET.WM"],
        param["REGNET.DEPTH"],
    )  # .values()#cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
    _, _, num_s, max_s, _, _ = generate_regnet(wa, w0, wm, d, 8)
    num_stages = NUM_STAGES
    if num_s != max_s or not num_stages[0] <= num_s <= num_stages[1]:
        return False
    return True


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


# STRIDE = 2
# BOT_MUL = 1.0
# GROUP_W_DEFAULT = 16
def generate_regnet_full(params):
    """Generates per stage ws, ds, gs, bs, and ss from RegNet cfg."""
    w_a, w_0, w_m, d = (
        params["REGNET.WA"],
        params["REGNET.W0"],
        params["REGNET.WM"],
        params["REGNET.DEPTH"],
    )
    ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
    ss = [2 for _ in ws]
    bs = [params["REGNET.BOT_MUL"] for _ in ws]
    gs = [params["REGNET.GROUP_W"] for _ in ws]
    ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
    return ws, ds, ss, bs, gs


" ws = widths , ds = depths, ss = stride, bs = bottlenet multipliers, gs = group w"


def sample_parameters():
    """Samples params [key, val, ...] list based on the samplers."""
    return regnet_sampler()


def sample_cfgs(seed, sample_size=32):
    """Samples chunk configs and return those that are unique and valid."""
    # Fix RNG seed (every call to this function should use a unique seed)
    np.random.seed(seed)
    # setup_cfg = sweep_cfg.SETUP
    cfgs = {}
    # sample_size = 32
    while True:
        # Sample parameters [key, val, ...] list based on the samplers
        params = sample_parameters()
        # Check if config is unique, if not continue
        key = params  # zip(params[0::2], params[1::2])
        key = " ".join(["{} {}".format(k, v) for k, v in key.items()])
        # if key in cfgs:
        #    continue
        # Generate config from parameters
        # reset_cfg()
        # cfg.merge_from_other_cfg(setup_cfg.BASE_CFG)
        # cfg.merge_from_list(params)
        # Check if config is valid, if not continue
        is_valid = check_regnet_constraints(params)
        if not is_valid:
            continue
        # Special logic for dealing w model scaling (side effect is to standardize cfg)
        # if cfg.MODEL.TYPE in ["anynet", "effnet", "regnet"]:
        #    scaler.scale_model()
        # only relavent to "Fast and Accurate Model Scaling".
        # For reference on scaling strategies, see: https://arxiv.org/abs/2103.06877.
        # Check if config is valid, if not continue
        # no way to calculate complexity, passed
        # is_valid = samplers.check_complexity_constraints(setup_cfg.CONSTRAINTS)
        # if not is_valid:
        #    continue
        # Set config description to key
        # cfg.DESC = key
        # Store copy of config if unique and valid
        cfgs[key] = params
        # Stop sampling if already reached quota
        if len(cfgs) >= sample_size:
            break
    return cfgs


if __name__ == "__main__":
    for i in range(10):
        sample = regnet_sampler()
        print(sample)
        results = generate_regnet_full(sample)
        print(results)
