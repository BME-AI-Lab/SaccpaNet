DEPTH = [12, 28]
W0 = [8, 256]
WA = [8.0, 256.0]
WM = [2.0, 3.0]
NUM_STAGES = [4, 4]
MIN_PARAMS = 4e6
MAX_PARAMS = 14e6


from .update_from_local_configs import update_from_local_configs

update_from_local_configs(globals())
