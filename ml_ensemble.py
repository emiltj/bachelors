import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import pandas as pd
import numpy as np

data1 = [pd.read_csv('./csv_files/train1.csv', index_col=0), 1, pd.read_csv('./csv_files/holdout1.csv')]
data2 = [pd.read_csv('./csv_files/train2.csv', index_col=0), 2, pd.read_csv('./csv_files/holdout2.csv')]
data3 = [pd.read_csv('./csv_files/train3.csv', index_col=0), 3, pd.read_csv('./csv_files/holdout3.csv')]
data4 = [pd.read_csv('./csv_files/train4.csv', index_col=0), 4, pd.read_csv('./csv_files/holdout4.csv')]
data5 = [pd.read_csv('./csv_files/train5.csv', index_col=0), 5, pd.read_csv('./csv_files/holdout5.csv')]

################## Creating a loop, so I won't have to do the below code 5 times ##################
# Making the datasets into a list
datasets = [data1, data2, data3, data4, data5]

#creating empty objects to save data in
classif_reports = ["", "", "", "", ""]
conf_mtxs = []
model = SVC(kernel = 'linear', class_weight = 'balanced')
SVM_coef = pd.DataFrame(columns = ['predictor_name', 'coef', 'fold'])

#defining which feature set is currently run 
feature_set = 'holdout_performance'

for train, n, holdout in datasets:   
    # Dividing train and holdout up into predictor variables, and Diagnosis
    x = train.iloc[:,4:]
    y = train.loc[:,['ID', 'Diagnosis']]
    y = y.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    x_holdout = holdout.iloc[:,4:]
    y_holdout = holdout.loc[:,['ID', 'Diagnosis']]
    y_holdout = y_holdout.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    # Fit the data onto the model
    model.fit(x, y)

    # Predict the holdout set (on the basis of predictor variables), using the fitted model
    y_predictions = model.predict(x_holdout)

    print(y_predictions)

    # Get out the coefficients
    coefs = model.coef_[0]
    
    # Get out the coefficient names
    coefs_names = list(x_holdout)
    coefs_names = np.asarray(coefs_names)

    # Create an array with info on which folds data/features we're handling!
    fold = np.repeat(n, len(coefs))

    # Load in the predictor names, and their coefficients to the empty dataframe
    SVM_coef = SVM_coef.append(pd.DataFrame({'predictor_name': coefs_names, 'coef' : coefs, 'fold' : fold}), ignore_index = True)

    # Get classification report
    report = classification_report(y_holdout, y_predictions, output_dict = True)
    report_number = n-1
    classif_reports[report_number] = report
    
    # Get confusion matrix
    matrixx = confusion_matrix(y_holdout, y_predictions)
    conf_mtxs.append(matrixx)
    classif_reports[4]

    #Saving data
   
    # making variables to use in filenames
    svm_coef_name = "".join(['./performance/holdout/', str(feature_set), "_coef_holdout_fold", ".csv"]) 
    conf_matrix_name = "".join(['./performance/holdout/', str(feature_set),"_confusion_matrix", str(n), ".csv"]) 
    classification_report_name = "".join(['./performance/holdout/', str(feature_set),"_classification_report", str(n), ".csv"]) 


    # Loading each element (classification_report) of the list of dictionaries, as a dataframe 
    classif_report_fold = pd.DataFrame(classif_reports[n-1])
    # Loading each element (classification_report) of the list of dictionaries, as a dataframe
    conf_matrix_fold = pd.DataFrame(conf_mtxs[n-1])
    #Give the matrices names for rows and columns
    conf_matrix_fold.columns = conf_matrix_fold.columns = ['predict_td', 'predict_sz']
    conf_matrix_fold.index = conf_matrix_fold.index = ['true_td', 'true_sz']

    # Writing coefficients to .csv
    SVM_coef.to_csv(svm_coef_name, sep=',', index = True)
    # Writing confusion matrices to csv
    conf_matrix_fold.to_csv(conf_matrix_name, sep=',', index = True)
    # Writing classification reports to csv
    classif_report_fold.to_csv(classification_report_name, sep=',', index = True)

    # The end

# Testing that the labels for the confusion matrices fit -> Yes they do!!!
# datasets[0][2].loc[datasets[0][2]['Diagnosis'] == 0]
# datasets[0][2].loc[datasets[0][2]['Diagnosis'] == 1]
# pd.read_csv("performance_measures_5_lassos/performance_measures_5_feature_sets_ConfusionMatrix1.csv")
# pd.read_csv("performance_measures_5_lassos/performance_measures_5_feature_sets_ClassificationReport1.csv")

pd.read_csv("performance/holdout/holdout_performance_classification_report1.csv")
pd.read_csv("performance/holdout/holdout_performance_classification_report2.csv")
pd.read_csv("performance/holdout/holdout_performance_classification_report3.csv")
pd.read_csv("performance/holdout/holdout_performance_classification_report4.csv")
pd.read_csv("performance/holdout/holdout_performance_classification_report5.csv")

pd.read_csv("performance/holdout/holdout_performance_confusion_matrix1.csv")
pd.read_csv("performance/holdout/holdout_performance_confusion_matrix2.csv")
pd.read_csv("performance/holdout/holdout_performance_confusion_matrix3.csv")
pd.read_csv("performance/holdout/holdout_performance_confusion_matrix4.csv")
pd.read_csv("performance/holdout/holdout_performance_confusion_matrix5.csv")

pd.read_csv("performance/holdout/holdout_performance_coef_holdout_fold.csv")







































































































# Old function with shitty results (but worked)
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import pandas as pd
import numpy as np
import os

# Load train and test data
data1 = [pd.read_csv('./csv_files/train1.csv', index_col=0), 1, pd.read_csv('./csv_files/holdout1.csv', index_col=0)]
data2 = [pd.read_csv('./csv_files/train2.csv', index_col=0), 2, pd.read_csv('./csv_files/holdout2.csv', index_col=0)]
data3 = [pd.read_csv('./csv_files/train3.csv', index_col=0), 3, pd.read_csv('./csv_files/holdout3.csv', index_col=0)]
data4 = [pd.read_csv('./csv_files/train4.csv', index_col=0), 4, pd.read_csv('./csv_files/holdout4.csv', index_col=0)]
data5 = [pd.read_csv('./csv_files/train5.csv', index_col=0), 5, pd.read_csv('./csv_files/holdout5.csv', index_col=0)]

# Take the lists and include them in a list
datasets = [data1, data2, data3, data4, data5]

# creating empty list of classif_reports - for testing!!!
classif_reports = ["", "", "", "", ""]

# Defining the model we're using
model = SVC(kernel = 'linear', class_weight = 'balanced')

# Preparing a dataframe for the predictions
predictions = pd.DataFrame({'ID' : datasets[0][2].loc[:,'ID'], 'sex' : datasets[0][2].loc[:, 'Gender'] , 'diagnosis_real' : datasets[0][2].loc[:,'Diagnosis']})

# Make sure diagnosis_real column also has 1's and 0's instead of 'td' and 'sz'
predictions['diagnosis_real'] = predictions['diagnosis_real'].apply(lambda x: 0 if x=='td' else 1)

for train, k, holdout in datasets:
    # making td into 0 and sz into 1 for both train and holdout
    train['Diagnosis'] = train['Diagnosis'].apply(lambda x: 0 if x=='td' else 1) # setting td as 0 and ASD as 1
    holdout['Diagnosis'] = holdout['Diagnosis'].apply(lambda x: 0 if x=='td' else 1) # setting td as 0 and ASD as 1

    # Cutting my training data up into x and y (so the model can be fit)
    x = train.iloc[:,4:]
    y = train.loc[:,['ID', 'Diagnosis']]
    y = y.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    # Cutting my testing data up into x and y (so the y's can be predicted)
    x_holdout = holdout.iloc[:,3:]
    y_holdout = holdout.loc[:,['ID', 'Diagnosis']]
    y_holdout = y_holdout.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    # Fit the data onto the model
    model.fit(x, y)

    # Predict the test set (on the basis of predictor variables), using the fitted model
    y_predicted = model.predict(x_holdout)
    
    # To test how it predicts!!
    print(y_predicted)
    
    # To test how they predict!!
    report = pd.DataFrame(classification_report(y_holdout, y_predicted, output_dict = True))
    report_number = k-1
    classif_reports[report_number] = report
    
    # ABOVE WORKS, THIS COULD BE ADDED TO LOOP:
    # Make a dataframe with ID and Real Diagnosis
    # predictions = predictions.append(pd.DataFrame({'ID' : holdout.loc[:,'ID'], 'diagnosis_real' : holdout.loc[:,'Diagnosis']}), ignore_index = True)
    # predictions = pd.DataFrame({'ID' : holdout.loc[:,'ID'], 'diagnosis_real' : holdout.loc[:,'Diagnosis']})
    #print(len(predictions['ID']))

    # Make a unique name for each iteration
    new_col_name = "".join(["diagnosis_predic_", str(k)])
    
    # Add a column to the dataframe (with unique name, and the predictions)
    predictions[new_col_name] = y_predicted

# For each row, get a count of 1's and 0's in the new columns
count_of_1_predictions = predictions.iloc[:,-5:].apply(pd.Series.value_counts, axis=1)[1].fillna(0)
predictions['diagnosis_predic_ensemble'] = [0 if x < 3 else 1 for x in count_of_1_predictions]

# Performance of ensemble_model on all participants
classification_report_ensemble = pd.DataFrame(classification_report(predictions['diagnosis_real'], predictions['diagnosis_predic_ensemble'], output_dict = True))
conf_matrix_ensemble = pd.DataFrame(confusion_matrix(predictions['diagnosis_real'], predictions['diagnosis_predic_ensemble']))
classification_report_ensemble
conf_matrix_ensemble

# Performance of ensemble_model on women
predictions_female = predictions[predictions['sex'] == 'F']
classification_report_ensemble_female = pd.DataFrame(classification_report(predictions_female['diagnosis_real'], predictions_female['diagnosis_predic_ensemble'], output_dict = True))
conf_matrix_ensemble_female = pd.DataFrame(confusion_matrix(predictions_female['diagnosis_real'], predictions_female['diagnosis_predic_ensemble']))

# Performance of ensemble_model on men
predictions_male = predictions[predictions['sex'] == 'M']
classification_report_ensemble_male = pd.DataFrame(classification_report(predictions_male['diagnosis_real'], predictions_male['diagnosis_predic_ensemble'], output_dict = True))
conf_matrix_ensemble_male = pd.DataFrame(confusion_matrix(predictions_male['diagnosis_real'], predictions_male['diagnosis_predic_ensemble']))

# Is the order shit? 


############# Results #############
# All predictions
predictions

# Performance of invididual submodels
classif_reports[0]
classif_reports[1]
classif_reports[2]
classif_reports[3]
classif_reports[4]

# Performance of ensemble
classification_report_ensemble
conf_matrix_ensemble

# Performance of ensemble for female
classification_report_ensemble_female
conf_matrix_ensemble_female

#Performance of ensemble for male
classification_report_ensemble_male
conf_matrix_ensemble_male

