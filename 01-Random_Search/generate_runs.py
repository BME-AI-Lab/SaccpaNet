import shutil


def copy_and_write(representation, params):
    shutil.copytree("codes", f"runs/{representation}", dirs_exist_ok=True)
    with open(f"runs/{representation}/config.py", "w") as f:
        f.write(params)


def search_range():
    for i in range(32):
        representation = f"seed_{i}"
        params = f"seed = {i}"
        copy_and_write(representation, params)


if __name__ == "__main__":
    search_range()
