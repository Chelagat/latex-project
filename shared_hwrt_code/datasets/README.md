## PROGRESS REPORT: CONVERTING HANDWRITTEN MATH EXPRESSIONS TO LATEX
# Overview
-Segmentation Baseline: cseg.py
-Character Recognition SVM: inkml.py
-Parser Baseline: inkml.py
-Train & Test Data: in baseline directory


# Instructions
- run `pip install -r requirements.txt` to install required python modules
- run `python inkml.py -s <first_directory_index> -e <last_directory_index>` to train OCR SVM, and to see parser baseline results, with input from the specified range of training data directories. E.g. `python inkml.py -s 1 -e 4` trains the SVM on all the training data. 
-run `python cseg.py` to see segmentation baseline results, (with input from all training data directories)
