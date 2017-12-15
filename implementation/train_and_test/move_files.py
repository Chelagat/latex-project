import os


folder = "/Users/norahborus/Documents/latex-project/baseline/training_data/TrainINKML_2013_JSON/"
new_folder = "/Users/norahborus/Documents/latex-project/baseline/training_data/TrainINKML_2013_JSON_Remainder/"

all_files = os.listdir(folder)
remainder = all_files[-444:]


for file in remainder:
    old_path = folder + file
    new_path  = new_folder + file
    os.rename(old_path, new_path)


print "DONE"


