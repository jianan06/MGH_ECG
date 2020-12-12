# MGH Internship research
Aim
Understand what respiration patterns and ECG patterns are different for different sleep stages.

What we have done
•	We have trained a deep learning model that achieves reasonable performance
•	We have defined ~8 features from respiration signals
•	We have fit decision trees using these features to stage sleep
•	We have fit decision trees using these features to predict the last hidden layer output of the deep network in each sleep stage

Things to do
brief literature review: using respiration and/or ECG to do sleep staging
Part I:
add better features:
•	load signals and sleep stages (.mat file) (mad3/Datasets_ConvertedData/sleeplab/natus_data AND grass_data)
•	find the channels (ECG and ABD)
•	for ABD: for each epoch, compute the features
•	for ECG: for each epoch, compute the features (features that are based R peaks)
