import shutil

params = {
    "REGNET.DEPTH": 19,
    "REGNET.W0": 40,
    "REGNET.WA": 24,
    "REGNET.WM": 2.209,
    "REGNET.GROUP_W": 40,
    "REGNET.BOT_MUL": 1,
}
DEPTH = 28


def copy_and_write(representation, params):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/config.py", "w") as f:
        f.write(params)


def set_up_search_range():
    result = list()
    for W0 in range(28, 48, 4):
        for WA in range(6, 10, 1):
            WM = 2.245
            p = params.copy()
            p.update(
                {
                    "REGNET.DEPTH": DEPTH,
                    "REGNET.W0": W0,
                    "REGNET.WA": WA,
                    "REGNET.WM": WM,
                }
            )
            representation = f"Depth{DEPTH}_W0{W0}_WA{WA}_WM{WM}"
            params = "params = " + repr(params)
            copy_and_write(representation, params)
            result.append((representation, p))
    return result


def search_range():
    for i in range(32):
        representation = f"seed_{i}"
        params = f"seed = {i}"
        copy_and_write(representation, params)


if __name__ == "__main__":
    search_range()
