import torch

x = torch.load("origin_model.pth")
y = torch.load("best_coco-wholebody_AP_epoch_210.pth")


def replace(d):
    """replace values of dict with its type, if and only if it is not an instace of dictionary

    Args:
        d (_type_): _description_
    """
    for i in d:
        if isinstance(d[i], dict):
            replace(d[i])
        elif isinstance(d[i], torch.Tensor):
            d[i] = d[i].shape
        else:
            d[i] = type(d[i])
    return d


def compare_and_replace(x, y):
    for key in x:
        if "backbone" in key:
            y_key = key.replace("backbone", "net.backbone")
        else:
            y_key = key
        if y_key not in y:
            print(f"{key} not in y")
            print()
            continue
        if isinstance(x[key], dict):
            compare_and_replace(x[key], y[y_key])
        elif isinstance(x[key], torch.Tensor):
            if x[key].shape != y[y_key].shape:
                print(f"Shape mismatch in {key}")
                print(x[key].shape)
                print(y[y_key].shape)
                print()
            else:
                y[y_key] = x[key]
    return y


# type_x = replace(x)
# type_y = replace(y["state_dict"])
merged = compare_and_replace(y["state_dict"], x)
torch.save(merged, "merged_model.pth")
