from tqdm.contrib.telegram import tqdm as _tqdm
from functools import partial
import os

token = os.environ.get("TELEGRAM_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")


def get_telegram_tqdm(token=None, chat_id=None):
    # if token is None:
    #     token = os.environ.get("TELEGRAM_TOKEN")
    #     # print(token)
    # if chat_id is None:
    #     chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    #     # print(chat_id)
    return partial(_tqdm, token=token, chat_id=chat_id)


# Progress bar hack

if token is not None and chat_id is not None:
    import tqdm.auto
    import tqdm

    tqdm.auto.tqdm = get_telegram_tqdm()
    tqdm.tqdm = get_telegram_tqdm()
    import pytorch_lightning.callbacks.progress.tqdm_progress as tqdm_progress

    _PAD_SIZE = 5

    class Tqdm(_tqdm):
        def __init__(self, *args, **kwargs) -> None:
            """Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from
            flickering."""
            # this just to make the make docs happy, otherwise it pulls docs which has some issues...
            token = os.environ.get("TELEGRAM_TOKEN")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID")
            super().__init__(
                token=token, chat_id=chat_id, min_interval=10, *args, **kwargs
            )

        @staticmethod
        def format_num(n):
            """Add additional padding to the formatted numbers."""
            should_be_padded = isinstance(n, (float, str))
            if not isinstance(n, str):
                n = _tqdm.format_num(n)
                assert isinstance(n, str)
            if should_be_padded and "e" not in n:
                if "." not in n and len(n) < _PAD_SIZE:
                    try:
                        _ = float(n)
                    except ValueError:
                        return n
                    n += "."
                n += "0" * (_PAD_SIZE - len(n))
            return n

    tqdm_progress.Tqdm = Tqdm
