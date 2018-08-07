import pickle
import os

file1 = "radscan_dx01_firstonly_dictionary_third.pkl"
file2 = "radscan_dx01_firstonly_manyepsilons_dictionary.pkl"

data1 = pickle.load(open(file1, "rb"))
data2 = pickle.load(open(file2, "rb"))

dictionary = {}

for element in data1:
	dictionary[element] = data1[element]

for element in data2:
	dictionary[element] = data2[element]

with open("radscan_dx01_firstonly_manyepsilons_dictionary_v2.pkl", "wb") as f:
	pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)