import os
import pickle
import warnings
from utils import get_string_file_size

'''
Script is used to combine two datasets into one
Data structure:
list(
    dict(
        "observations": nparray(nparray(np.float32)),
        "next_observations": nparray(nparray(np.float32)),
        "actions": nparray(nparray(np.float32)),
        "rewards": nparray(np.oat32),
        "terminals": nparray(np.bool_)
        )
    )
'''

if __name__ == '__main__':

    dataset1 = "./data/L_R1_2189x5x8.pkl"
    dataset2 = "./data/R2_2189x5x4.pkl"

    file_prefix = "L_R1_R2"

    amount_buildings = 5

    with open(dataset1, "rb") as d1:
        data1 = pickle.load(d1)  # list

        length1 = len(data1[0]["observations"])
        for d in data1:
            if len(d["observations"]) != length1:
                warnings.warn("Sequences in dataset1 are not all the same length")
        print("Dataset1:")
        print("Amount Of Sequences:", len(data1), " Size:", get_string_file_size(dataset1))

        with open(dataset2, "rb") as d2:
            data2 = pickle.load(d2)  # list

            length2 = len(data2[0]["observations"])
            for d in data2:
                if len(d["observations"]) != length2:
                    warnings.warn("Sequences in dataset2 are not all the same length")
            print("Dataset2:")
            print("Amount Of Sequences:", len(data2), " Size:", get_string_file_size(dataset2))

            if length1 != length2:
                warnings.warn("Sequences in dataset1 and dataset2 are not same length")
            else:
                print("Sequence length of both datasets:", length1, length2)

            ''' MERGE DATASETS '''
            merged_data = data1 + data2
            length_merged_data = len(merged_data)

            if length_merged_data != len(data1) + len(data2):
                warnings.warn(str(length_merged_data) + "!=" + str(len(data1)) + "+" + str(len(data2)))

            ''' Format: file_prefix _ [SEQUENCE_LENGTH] x [AMOUNT_BUILDINGS] x [AMOUNT_EPISODES] '''
            file_info = "_" + str(max(length1, length2)) + "x" + str(amount_buildings) + "x" + str(int(length_merged_data/amount_buildings))
            file_extension = ".pkl"
            file_name = file_prefix + file_info + file_extension
            file_path = "./data/" + file_name

            # create or overwrite pickle file
            with open(file_path, "wb") as f:
                pickle.dump(merged_data, f)

            print("========================= Merging Completed ============================")
            print("==> Data saved in", file_name, get_string_file_size(file_path))


