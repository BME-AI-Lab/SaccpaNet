lr = 2.5e-5  # 1e-4#0.02#0.00002c
l2 = 0  # 5e-8


from .update_from_local_configs import update_from_local_configs

update_from_local_configs(globals())
