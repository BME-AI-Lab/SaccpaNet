# %%
from configs import dataset_config

old_dataset_path = dataset_config.SQLITE_DB_PATH
new_dataset_path = dataset_config.FLIPCUT_AUGMENTED_SQLITE_DB_PATH
new_dataset_db_connection_string = dataset_config.FLIPCUT_AUGMENTED_DB_CONNECTION_STRING

import pandas as pd

annotation_df = pd.read_sql_query(
    "SELECT * FROM annotations", dataset_config.DB_CONNECTION_STRING
)
image_df = pd.read_sql_query(
    "SELECT * FROM depth_images", dataset_config.DB_CONNECTION_STRING
)
non_quilted_annotation_df = annotation_df[annotation_df["effect"] == "4"]

annotation_df.drop(columns=["level_0"], inplace=True)
image_df.drop(columns=["level_0"], inplace=True)

new_annotation_df = annotation_df.copy()
new_image_df = image_df.copy()


#

new_effect = [
    "topCovered_downUncovered",
    "topUncovered_downCovered",
    "leftCovered_rightUncovered",
    "leftUncovered_rightCovered",
]


# Definition of effects:
# 1: Thick
# 2: Medium
# 3: Thin
# 4: None
# 5: 1_topCovered_downUncovered
# 6: 1_topUncovered_downCovered
# 7: 1_leftCovered_rightUncovered
# 8: 1_leftUncovered_rightCovered
# 9: 2_topCovered_downUncovered
# 10: 2_topUncovered_downCovered
# 11: 2_leftCovered_rightUncovered
# 12: 2_leftUncovered_rightCovered
# 13: 3_topCovered_downUncovered
# 14: 3_topUncovered_downCovered
# 15: 3_leftCovered_rightUncovered
# 16: 3_leftUncovered_rightCovered

# top_bottom refers to the bed axis, and therefore x axis
# left_right refers y axis
# %%
import pickle as pkl

import tqdm

max_id = image_df["index"].max()
additional_image_rows = []
additional_annotation_rows = []
is_4_skiped = False
for index, row in tqdm.tqdm(annotation_df.iterrows(), total=len(annotation_df)):
    if row["effect"] == "4":  # No need to generate image for no quilt
        if not is_4_skiped:
            print("Skipping effect 4 (no quilt)")
            is_4_skiped = True
        continue
    no_quilt_annotation = non_quilted_annotation_df[
        non_quilted_annotation_df["image_id"] == row["image_id"]
    ]
    assert len(no_quilt_annotation) == 1
    no_quilt_annotation = no_quilt_annotation.iloc[0]
    no_quilt_image_row = image_df[image_df["index"] == no_quilt_annotation["index"]]
    no_quilt_image_depth_array = no_quilt_image_row["depth_array"].values[0]
    no_quilt_image_depth_array = pkl.loads(no_quilt_image_depth_array)
    no_quilt_image_depth_color_frame = no_quilt_image_row["depth_color_frame"].values[0]
    no_quilt_image_depth_color_frame = pkl.loads(no_quilt_image_depth_color_frame)
    no_quilt_color_frame_image_align = no_quilt_image_row[
        "color_frame_image_align"
    ].values[0]
    no_quilt_color_frame_image_align = pkl.loads(no_quilt_color_frame_image_align)

    x1, y1, x2, y2 = (row["x1"], row["y1"], row["x2"], row["y2"])
    left_right_cut_pt = int((x2 - x1) / 2 + x1)
    top_bottom_cut_pt = int((y2 - y1) / 2 + y1)
    # print(f"left_right_cut_pt: {left_right_cut_pt}, top_bottom_cut_pt: {top_bottom_cut_pt}")
    # print(no_quilt_image_depth_array.shape)
    # break

    for effect in new_effect:
        new_row = row.copy()
        max_id += 1
        new_row["index"] = int(max_id)
        new_row["effect"] = row["effect"] + "_" + effect
        additional_annotation_rows.append(pd.DataFrame(new_row).T)
        image_row = image_df[image_df["index"] == row["index"]]
        new_image_row = image_row.copy()
        new_image_row["index"] = int(max_id)
        if effect == "leftCovered_rightUncovered":
            depth_frame = new_image_row["depth_array"].values[0]
            depth_frame = pkl.loads(depth_frame)
            depth_frame[:top_bottom_cut_pt, :] = no_quilt_image_depth_array[
                :top_bottom_cut_pt, :
            ]
            depth_frame = pkl.dumps(depth_frame)
            new_image_row["depth_array"] = depth_frame
            depth_color_frame = new_image_row["depth_color_frame"].values[0]
            depth_color_frame = pkl.loads(depth_color_frame)
            depth_color_frame[:top_bottom_cut_pt, :] = no_quilt_image_depth_color_frame[
                :top_bottom_cut_pt, :
            ]
            depth_color_frame = pkl.dumps(depth_color_frame)
            new_image_row["depth_color_frame"] = depth_color_frame
            color_frame_image_align = new_image_row["color_frame_image_align"].values[0]
            color_frame_image_align = pkl.loads(color_frame_image_align)
            color_frame_image_align[:top_bottom_cut_pt, :] = (
                no_quilt_color_frame_image_align[:top_bottom_cut_pt, :]
            )
            color_frame_image_align = pkl.dumps(color_frame_image_align)
            new_image_row["color_frame_image_align"] = color_frame_image_align
        elif effect == "leftUncovered_rightCovered":
            depth_frame = new_image_row["depth_array"].values[0]
            depth_frame = pkl.loads(depth_frame)
            depth_frame[top_bottom_cut_pt:, :] = no_quilt_image_depth_array[
                top_bottom_cut_pt:, :
            ]
            depth_frame = pkl.dumps(depth_frame)
            new_image_row["depth_array"] = depth_frame
            depth_color_frame = new_image_row["depth_color_frame"].values[0]
            depth_color_frame = pkl.loads(depth_color_frame)
            depth_color_frame[top_bottom_cut_pt:, :] = no_quilt_image_depth_color_frame[
                top_bottom_cut_pt:, :
            ]
            depth_color_frame = pkl.dumps(depth_color_frame)
            new_image_row["depth_color_frame"] = depth_color_frame
            color_frame_image_align = new_image_row["color_frame_image_align"].values[0]
            color_frame_image_align = pkl.loads(color_frame_image_align)
            color_frame_image_align[top_bottom_cut_pt:, :] = (
                no_quilt_color_frame_image_align[top_bottom_cut_pt:, :]
            )
            color_frame_image_align = pkl.dumps(color_frame_image_align)
            new_image_row["color_frame_image_align"] = color_frame_image_align
        elif effect == "topUncovered_downCovered":
            depth_frame = new_image_row["depth_array"].values[0]
            depth_frame = pkl.loads(depth_frame)
            depth_frame[:, :left_right_cut_pt] = no_quilt_image_depth_array[
                :, :left_right_cut_pt
            ]
            depth_frame = pkl.dumps(depth_frame)
            new_image_row["depth_array"] = depth_frame
            depth_color_frame = new_image_row["depth_color_frame"].values[0]
            depth_color_frame = pkl.loads(depth_color_frame)
            depth_color_frame[:, :left_right_cut_pt] = no_quilt_image_depth_color_frame[
                :, :left_right_cut_pt
            ]
            depth_color_frame = pkl.dumps(depth_color_frame)
            new_image_row["depth_color_frame"] = depth_color_frame
            color_frame_image_align = new_image_row["color_frame_image_align"].values[0]
            color_frame_image_align = pkl.loads(color_frame_image_align)
            color_frame_image_align[:, :left_right_cut_pt] = (
                no_quilt_color_frame_image_align[:, :left_right_cut_pt]
            )
            color_frame_image_align = pkl.dumps(color_frame_image_align)
            new_image_row["color_frame_image_align"] = color_frame_image_align
        elif effect == "topCovered_downUncovered":
            depth_frame = new_image_row["depth_array"].values[0]
            depth_frame = pkl.loads(depth_frame)
            depth_frame[:, left_right_cut_pt:] = no_quilt_image_depth_array[
                :, left_right_cut_pt:
            ]
            depth_frame = pkl.dumps(depth_frame)
            new_image_row["depth_array"] = depth_frame
            depth_color_frame = new_image_row["depth_color_frame"].values[0]
            depth_color_frame = pkl.loads(depth_color_frame)
            depth_color_frame[:, left_right_cut_pt:] = no_quilt_image_depth_color_frame[
                :, left_right_cut_pt:
            ]
            depth_color_frame = pkl.dumps(depth_color_frame)
            new_image_row["depth_color_frame"] = depth_color_frame
            color_frame_image_align = new_image_row["color_frame_image_align"].values[0]
            color_frame_image_align = pkl.loads(color_frame_image_align)
            color_frame_image_align[:, left_right_cut_pt:] = (
                no_quilt_color_frame_image_align[:, left_right_cut_pt:]
            )
            color_frame_image_align = pkl.dumps(color_frame_image_align)
            new_image_row["color_frame_image_align"] = color_frame_image_align

        additional_image_rows.append(pd.DataFrame(new_image_row))
# %%
del annotation_df
del image_df
import gc

gc.collect()
# %%
additional_annotation_rows = pd.concat(
    [i for i in additional_annotation_rows], copy=False
)
new_annotation_df = pd.concat(
    [new_annotation_df, additional_annotation_rows], copy=False
)
# del additional_annotation_rows
additional_image_rows = pd.concat([i for i in additional_image_rows], copy=False)
new_image_df = pd.concat([new_image_df, additional_image_rows], copy=False)
# del additional_image_rows


# %%
new_annotation_df.to_sql(
    "annotations", new_dataset_db_connection_string, if_exists="replace"
)
new_image_df.to_sql(
    "depth_images", new_dataset_db_connection_string, if_exists="replace"
)

# #%%
# map = {
#     "leftCovered_rightUncovered": "leftUncovered_rightCovered",
#     "leftUncovered_rightCovered": "leftCovered_rightUncovered",
# }
# map2 = {}
# for i in range(4):
#     for key,value in map.items():
#         map2[f"{i}_{key}"] = f"{i}_{value}"


# # %%
