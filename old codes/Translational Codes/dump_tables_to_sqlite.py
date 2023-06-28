# %%
import sqlite3

import pandas as pd

DB_CONNECTION_STRING = "mysql+pymysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu"

TABLE_NAME = "openpose_annotation_03_18_with_quilt"

connection = sqlite3.connect("dataset.db")
original_db = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", DB_CONNECTION_STRING)
# filter out the hold out data
filtered_db = original_db[original_db["subset"] != "hold"]
# filter out unnecessary columns
filtered_db = filtered_db.drop(
    columns=["mean_confidence"]  # , "annotation_file", "source_file", "index"]
)

# rename the subset to train, val, test
name_map = {"train": "train", "test": "val", "new_test": "test"}
filtered_db["subset"] = filtered_db["subset"].apply(lambda x: name_map[x])
# Note: remember to change the subset name in the dataset class accordingly

# print the number of rows to check
print("Total number of rows: ", len(filtered_db))
print("Number of rows in train: ", len(filtered_db[filtered_db["subset"] == "train"]))
print("Number of rows in val: ", len(filtered_db[filtered_db["subset"] == "val"]))
print("Number of rows in test: ", len(filtered_db[filtered_db["subset"] == "test"]))


filtered_db.to_sql(
    "annotations",
    connection,
    if_exists="replace",
)

# save the image data to a separate table
image_db = pd.read_sql_query(
    """
    select `a`.`index` AS `index`,`a`.`depth_array` AS `depth_array`,`a`.`depth_color_frame` AS `depth_color_frame`,`a`.`color_frame_image` AS `color_frame_image`,`a`.`color_frame_image_align` AS `color_frame_image_align`
    from (`bmepolyu`.`data_06_02` `a`
    join `bmepolyu`.`bag_subject` `c`
    on(`a`.`source_file` = `c`.`bag_file`))
    where `c`.`subset` != 'hold'
    """,
    DB_CONNECTION_STRING,
)
image_db.to_sql("depth_images", connection, if_exists="replace")
# %%
connection.close()
