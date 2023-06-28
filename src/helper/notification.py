import os
from functools import partial

from tqdm.contrib.telegram import tqdm as _tqdm


def get_telegram_tqdm(token=None, chat_id=None):
    if token is None:
        token = os.environ.get("TELEGRAM_TOKEN")
        print(token)
    if chat_id is None:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        print(chat_id)
    return partial(_tqdm, token=token, chat_id=chat_id)
