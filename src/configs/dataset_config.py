import os

DATALOADER_WORKERS = 4
SQLITE_DB_PATH = "..\\dataset.db"
DB_CONNECTION_STRING = f"sqlite:///{SQLITE_DB_PATH}"

if os.environ.get("SQLITE_DB_PATH"):
    SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH")
    DB_CONNECTION_STRING = f"sqlite:///{SQLITE_DB_PATH}"

from .update_from_local_configs import update_from_local_configs

update_from_local_configs(globals())
