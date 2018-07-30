import os
import pickle
import numpy as np

directories = ["data_summary_H/", "data_summary_HV/", "data_summary_V/"]

dictionary = {}

for directory in directories:
	files = os.listdir(directory)
	data_corrected1 = []
	data_uncorrected1 = []
	data_corrected2 = []
	data_uncorrected2 = []
	for file in files:
		if "Ron" in file:
			if "B1" in file:
				data_corrected1.append(file)
			else:
				data_corrected2.append(file)
		else:
			if "B1" in file:
				data_uncorrected1.append(file)
			else:
				data_uncorrected2.append(file)
	
	DAs_corrected1 = []
	DAs_uncorrected1 = []
	
	DAs_corrected2 = []
	DAs_uncorrected2 = []

	for filename in data_corrected1:
		temp_DA = {}
		file = open(directory+filename, 'r')
		for line in file:
			temp_DA[int(line.split(" ")[16 - 1])] = float(line.split(" ")[8-1])
		DAs_corrected1.append(temp_DA)
	
	for filename in data_uncorrected1:
		temp_DA = {}
		file = open(directory+filename, 'r')
		for line in file:
			temp_DA[int(line.split(" ")[16-1])] = float(line.split(" ")[8-1])
		DAs_uncorrected1.append(temp_DA)

	for filename in data_corrected2:
		temp_DA = {}
		file = open(directory+filename, 'r')
		for line in file:
			temp_DA[int(line.split(" ")[16 - 1])] = float(line.split(" ")[8-1])
		DAs_corrected2.append(temp_DA)
	
	for filename in data_uncorrected2:
		temp_DA = {}
		file = open(directory+filename, 'r')
		for line in file:
			temp_DA[int(line.split(" ")[16-1])] = float(line.split(" ")[8-1])
		DAs_uncorrected2.append(temp_DA)

	temp_dict = {}
	temp_dict ["cor1"] = DAs_corrected1
	temp_dict ["cor2"] = DAs_corrected2
	temp_dict ["uncor1"] = DAs_uncorrected1
	temp_dict ["uncor2"] = DAs_uncorrected2

	dictionary[directory[13:-1]] = temp_dict

with open("LHC_DATA.pkl", "wb") as f:
	pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)