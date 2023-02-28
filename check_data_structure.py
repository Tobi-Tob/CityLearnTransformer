import os
import pickle

'''
Script is used to visualize structure and dimension of a pickle data file
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

    file = "s_random.pkl"
    with open(file, "rb") as f:
        data = pickle.load(f)

        print("data ", type(data), "length: ", len(data))
        dict0 = data[0]
        print("data[0] ", type(dict0), "length: ", len(dict0))
        print("data[0].keys() ", dict0.keys())
        print()
        observations = dict0['observations']
        print("observations ", type(observations), "length: ", len(observations))
        observations0 = observations[0]
        print("observations[0] ", type(observations0), "length: ", len(observations0))
        observations00 = observations[0][0]
        print("observations[0][0] ", type(observations00))
        print()
        next_observations = dict0['next_observations']
        print("next_observations ", type(next_observations), "length: ", len(next_observations))
        next_observations0 = next_observations[0]
        print("next_observations[0] ", type(next_observations0), "length: ", len(next_observations0))
        next_observations00 = next_observations[0][0]
        print("next_observations[0][0] ", type(next_observations00))
        print()
        actions = dict0['actions']
        print("actions ", type(actions), "length: ", len(actions))
        actions0 = actions[0]
        print("actions[0] ", type(actions0), "length: ", len(actions0))
        actions00 = actions[0][0]
        print("actions[0][0] ", type(actions00))
        print()
        rewards = dict0['rewards']
        print("rewards ", type(rewards), "length: ", len(rewards))
        rewards0 = rewards[0]
        print("rewards[0] ", type(rewards0))
        print()
        terminals = dict0['terminals']
        print("terminals ", type(terminals), "length: ", len(terminals))
        terminals0 = terminals[0]
        print("terminals[0] ", type(terminals0))
        print()
        print("========================= Data Size =============================")
        length = 0
        for d in data:
            if len(d["observations"]) > length:
                length = len(d["observations"])

        print("Amount Of Sequences: ", len(data))
        print("Longest Sequence: ", length)

        file_size = os.stat(file).st_size
        if file_size > 1e+6:
            string_byte = "(" + str(round(file_size / 1e+6)) + " MB)"
        else:
            string_byte = "(" + str(round(file_size / 1e+3)) + " kB)"
        print(file, string_byte)

        # print(data[0]["observations"][0][:3])
