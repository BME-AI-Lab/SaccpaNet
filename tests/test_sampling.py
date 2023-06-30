from lib.modules.core.rand import plot_rand_cdfs
from lib.modules.core.sampler import generate_regnet_full, sample_cfgs


def test_sampler():
    keys = ["REGNET.WA", "REGNET.W0", "REGNET.WM", "REGNET.DEPTH"]
    SAMPLES = sample_cfgs(seed=0, sample_size=10)
    for PARAM_NAME, params in SAMPLES.items():
        for key in keys:
            assert key in params
        ws, ds, ss, bs, gs = generate_regnet_full(params)
        assert len(ws) == len(ds)
        assert len(ws) == 4


def test_distribution():
    import matplotlib

    matplotlib.use("Agg")
    plot_rand_cdfs()
