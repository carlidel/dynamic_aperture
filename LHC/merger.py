import os
import pickle

directories = ["data_summary_H/", "data_summary_HV/", "data_summary_V/"]

for directory in directories:
	files = os.listdir(directory)
	data_corrected = []
	data_uncorrected = []
	for file in files:
		if "Ron" in file:
			data_corrected.append(file)
		else:
			data_uncorrected.append(file)
	