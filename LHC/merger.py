import os
import pickle

directories = ["data_summary_H/", "data_summary_HV/", "data_summary_V/"]

dictionary = {}

for directory in directories:
	files = os.listdir(directory)
	data_corrected = []
	data_uncorrected = []
	for file in files:
		if "Ron" in file:
			data_corrected.append(file)
		else:
			data_uncorrected.append(file)
	
	times_corrected = []
	DAs_corrected = []
	times_uncorrected = []
	DAs_uncorrected = []

	for filename in data_corrected:
		temp_time = []
		temp_DA = []
		file = open(directory+filename, 'r')
		for line in file:
			temp_DA.append(float(line.split(" ")[8-1]))
			temp_time.append(int(line.split(" ")[16-1]))
		times_corrected.append(temp_time)
		DAs_corrected.append(temp_DA)
	
	for filename in data_corrected:
		temp_time = []
		temp_DA = []
		file = open(directory+filename, 'r')
		for line in file:
			temp_DA.append(float(line.split(" ")[8-1]))
			temp_time.append(int(line.split(" ")[16-1]))
		times_uncorrected.append(temp_time)
		DAs_uncorrected.append(temp_DA)

	dictionary[directory[13:-1]] = (times_corrected, DAs_corrected, times_uncorrected, DAs_uncorrected)

with open("LHC_DATA.pkl", "wb") as f:
	pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)