params = {
    "REGNET.DEPTH": 28,
    "REGNET.W0": 104,
    "REGNET.WA": 35.7,
    "REGNET.WM": 2,
    "REGNET.GROUP_W": 40,
    "REGNET.BOT_MUL": 1,
}

from .update_from_local_configs import update_from_local_configs

update_from_local_configs(globals())
