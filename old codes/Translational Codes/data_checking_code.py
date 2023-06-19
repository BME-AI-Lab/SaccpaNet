# check the coordinate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

error_set = set()


def load_annotations_by_data(postures):
    database_string = "sqlite:///D:\\Posture Coordinate Models\\dataset.db"

    df = pd.read_sql_query("SELECT * FROM annotations", database_string)
    df = df[(df["posture"] == postures) & (df["effect"] == "1")]
    return df


for postures in ["r", "o", "y", "g", "c", "b", "p"]:
    df = load_annotations_by_data(postures)
    num_joints = 18

    missing_joints = {i: set() for i in range(num_joints)}
    array = np.zeros((len(df), num_joints, 2))
    for subject, (index, row) in enumerate(df.iterrows()):
        for joint in range(num_joints):
            x, y = float(row[f"{joint}_x"]), float(row[f"{joint}_y"])
            array[subject, joint, :] = [x, y]
            if x == 0 or y == 0:
                image_id = row["image_id"]
                missing_joints[joint] = missing_joints[joint].union({image_id})
    # for i in range(num_joints):
    #     if any(missing_joints[i]):
    #         print(i, missing_joints[i])

    # skeleton = [
    #     [16, 14],
    #     [14, 12],
    #     [17, 15],
    #     [15, 13],
    #     [12, 13],
    #     [6, 12],
    #     [7, 13],
    #     [6, 7],
    #     [6, 8],
    #     [7, 9],
    #     [8, 10],
    #     [9, 11],
    #     [2, 3],
    #     [1, 2],
    #     [1, 3],
    #     [2, 4],
    #     [3, 5],
    #     [4, 6],
    #     [5, 7],
    # ]

    # normalize the distance by the distance between the head and the pelvis
    # norm_dist = (
    #     np.linalg.norm(array[:, 1, :] - array[:, 8, :], axis=1)
    #     + np.linalg.norm(array[:, 1, :] - array[:, 11, :], axis=1)
    # ) / 2

    # skeleton = [
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [1, 5],
    #     [5, 6],
    #     [6, 7],
    #     [1, 8],
    #     [8, 9],
    #     [9, 10],
    #     [1, 11],
    #     [11, 12],
    #     [12, 13],
    #     [1, 0],
    #     [0, 14],
    #     [14, 16],
    #     [0, 15],
    #     [15, 17],
    # ]
    # # additional plane for comparisons
    # skeleton += [[2, 8], [5, 11], [0, 17], [0, 16], [17, 16]]
    # error_set = set()
    # display_dict = {}
    # for i in skeleton:
    #     x, y = i
    #     # x, y = x - 1, y - 1
    #     distance = np.linalg.norm(array[:, x, :] - array[:, y, :], axis=1) / norm_dist
    #     display_dict[tuple(i)] = distance
    #     # sns.distplot(distance)
    #     # plt.show()
    #     # subjects > 3 sigma
    #     more_than = np.where(distance > distance.mean() + 3 * distance.std())
    #     less_than = np.where(distance < distance.mean() - 3 * distance.std())
    #     # print(f"{i} More than 3 sigma: {more_than}, {distance[more_than]}")
    #     # print(f"{i} Less than 3 sigma: {less_than}, {distance[less_than]}")

    #     subjects_more_than = set(df["image_id"].iloc[more_than])
    #     if any(subjects_more_than):
    #         print(i, subjects_more_than, distance[more_than])
    #     # print(i, subjects_more_than, distance[more_than])
    #     subjects_less_than = set(df["image_id"].iloc[less_than])
    #     if any(subjects_less_than):
    #         print(i, subjects_less_than, distance[less_than])
    #     # print(i, subjects_less_than, distance[less_than])
    #     error_set = error_set.union(subjects_more_than)
    #     error_set = error_set.union(subjects_less_than)

    # print(f"total error of {postures}", list(error_set))

# stricter assertion on some specific joints
# Legs to hip ratio

# calculating for bone angles
# adjacency_table = {i: set() for i in range(18)}
# for i, j in skeleton:
#     adjacency_table[i].add(j)
#     adjacency_table[j].add(i)
# for I, js in adjacency_table.items():
#     adjacency_table[i] = list(adjacency_table[i])
#     # select any two point
#     combinations_list = list(combinations(js, 2))
#     for J, K in combinations_list:
#         i, j, k = I - 1, J - 1, K - 1
#         Is, Js, Ks = array[:, i, :], array[:, j, :], array[:, k, :]
#         ij = Js - Is
#         ik = Ks - Is
#         dots = np.array([np.dot(ij[i], ik[i]) for i in range(len(ij))])
#         cosine_angle = dots / (np.linalg.norm(ij, axis=1) * np.linalg.norm(ik, axis=1))
#         angle = np.array([np.arccos(cosine_angle[i]) for i in range(len(cosine_angle))])
#         degress = np.degrees(angle)
#         print(
#             degress.shape,
#             Is.shape,
#             Js.shape,
#             Ks.shape,
#             ij.shape,
#             ik.shape,
#             dots.shape,
#             cosine_angle.shape,
#             angle.shape,
#         )
#         sns.distplot(degress)
#         plt.show()
#         print(j, i, k)
#         more_than = np.where(degress > degress.mean() + 3 * degress.std())
#         less_than = np.where(degress < degress.mean() - 3 * degress.std())
#         print(f"More than 3 sigma: {more_than}")
#         print(f"Less than 3 sigma: {less_than}")
#         subjects_more_than = set(df["subject_number"].iloc[more_than])
#         # print(subjects_more_than)
#         subjects_less_than = set(df["subject_number"].iloc[less_than])
#         # print(subjects_less_than)
#         error_set = error_set.union(subjects_more_than)
#         error_set = error_set.union(subjects_less_than)


# print(adjacency_table)
