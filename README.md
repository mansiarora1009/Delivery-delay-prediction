# Delivery Delay Prediction

**Data**
This folder contains files for the training and test data that you can use to run the models

**Final Output**
This folder contains the final output file with binary predictions for the final test data

**Parameter Tuning Results**
This folder contains intermediate files (not required for any scripts to run) with parameter tuning results for some clusters

**Model Files**
This folder contains the train and test scripts, data, saved model files and saved scaler files.
1. To test the latest data file, run the Test_Model.py script – This script will load the models and scalers already pretrained using the Train_Model.py script
2. The test script will generate the output csv file with binary delay values
3. Please make sure the directory structure is correct
The Train_Model script trains the model along with parameter tuning and might take around 20 minutes to run. The Test_Model script will take less than 20 seconds to run for the current test script.
The train and test files are provided in the submission.
