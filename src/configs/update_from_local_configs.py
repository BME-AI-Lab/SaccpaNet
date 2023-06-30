try:
    import local_config

    def update_from_local_configs(_global):
        for key, value in local_config.__dict__.items():
            if key in _global:
                _global[key] = value
        return

except ImportError:

    def update_from_local_configs(_global):
        print("No local config found")
        return
