import pandas as pd

def get_image_list():
    TABLE = "SELECT * from openpose_annotation_06_02 where subset='train' or subset='test'"
    CONNECTION_STRING = "mysql+pymysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu"
    df = pd.read_sql_query(TABLE,CONNECTION_STRING)
    indexs = []
    joints = []
    list_of_joints_to_be_fixed = range(1,14)
    for i in list_of_joints_to_be_fixed:
        not_confident = df[f"{i}_confidence"] < 0.5
        l = list(df[not_confident][f"image_id"])
        indexs += l
        joints += [i]*len(l)
    return zip(indexs, joints)

#image 96 is strange
#image 16 fliped
#656 fliped

if __name__ == "__main__":
    for i,j in get_image_list():
        print(i,j)