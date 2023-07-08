params = {
    "REGNET.DEPTH": 17,
    "REGNET.W0": 8,
    "REGNET.WA": 12.8,
    "REGNET.WM": 2.942,
}

from .update_from_local_configs import update_from_local_configs

update_from_local_configs(globals())
