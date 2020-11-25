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
predictions = pd.DataFrame({'ID' : datasets[0][2].loc[:,'ID'], 'sex' : datasets[0][2].loc[:, 'Gender'] , 'diagnosis_real' : datasets[0][2].loc[:,'Diagnosis']})

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

    # Make a unique name for each iteration
    new_col_name = "".join(["diagnosis_predic_", str(n)])
    
    # Add a column to the dataframe (with unique name, and the predictions)
    predictions[new_col_name] = y_predictions

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

    # Saving data
    # making variables to use in filenames
    svm_coef_name = "".join(['./performance/holdout/', "coefs_all_models", ".csv"]) 
    conf_matrix_name = "".join(['./performance/holdout/models/', "confusion_matrix", str(n), ".csv"]) 
    classification_report_name = "".join(['./performance/holdout/models/', "classification_report", str(n), ".csv"]) 

    # Loading each element (classification_report) of the list of dictionaries, as a dataframe 
    classif_report_fold = pd.DataFrame(classif_reports[n-1])
    # Loading each element (classification_report) of the list of dictionaries, as a dataframe
    conf_matrix_fold = pd.DataFrame(conf_mtxs[n-1])
    # Give the matrices names for rows and columns
    conf_matrix_fold.columns = conf_matrix_fold.columns = ['predict_td', 'predict_sz']
    conf_matrix_fold.index = conf_matrix_fold.index = ['true_td', 'true_sz']

    # Writing coefficients to .csv
    SVM_coef.to_csv(svm_coef_name, sep=',', index = True)
    # Writing confusion matrices to csv
    conf_matrix_fold.to_csv(conf_matrix_name, sep=',', index = True)
    # Writing classification reports to csv
    classif_report_fold.to_csv(classification_report_name, sep=',', index = True)
    # The end

# Fixing "predictions" dataframe
count_of_1_predictions = predictions.iloc[:,-5:].apply(pd.Series.value_counts, axis=1)[1].fillna(0)
predictions['diagnosis_predic_ensemble'] = [0 if x < 3 else 1 for x in count_of_1_predictions] # For each row, get a count of 1's and 0's in the new columns

# Performance of ensemble all
classification_report_ensemble = pd.DataFrame(classification_report(predictions['diagnosis_real'], predictions['diagnosis_predic_ensemble'], output_dict = True))
conf_matrix_ensemble = pd.DataFrame(confusion_matrix(predictions['diagnosis_real'], predictions['diagnosis_predic_ensemble']))
classification_report_ensemble.to_csv('./performance/holdout/ensemble/classification_report.csv', sep = ",", index = True)
conf_matrix_ensemble.to_csv('./performance/holdout/ensemble/confusion_matrix.csv', sep = ",", index = True)
predictions.to_csv('./performance/holdout/all_predictions_holdout.csv')

# Performance of ensemble female
predictions_female = predictions[predictions['sex'] == 'F']
classification_report_ensemble_female = pd.DataFrame(classification_report(predictions_female['diagnosis_real'], predictions_female['diagnosis_predic_ensemble'], output_dict = True))
conf_matrix_ensemble_female = pd.DataFrame(confusion_matrix(predictions_female['diagnosis_real'], predictions_female['diagnosis_predic_ensemble']))
classification_report_ensemble_female.to_csv('./performance/holdout/ensemble/sex/female_classification_report.csv', sep = ",", index = True)
conf_matrix_ensemble_female.to_csv('./performance/holdout/ensemble/sex/female_confusion_matrix.csv', sep = ",", index = True)

# Performance of ensemble male
predictions_male = predictions[predictions['sex'] == 'M']
classification_report_ensemble_male = pd.DataFrame(classification_report(predictions_male['diagnosis_real'], predictions_male['diagnosis_predic_ensemble'], output_dict = True))
conf_matrix_ensemble_male = pd.DataFrame(confusion_matrix(predictions_male['diagnosis_real'], predictions_male['diagnosis_predic_ensemble']))
classification_report_ensemble_male.to_csv('./performance/holdout/ensemble/sex/male_classification_report.csv', sep = ",", index = True)
conf_matrix_ensemble_male.to_csv('./performance/holdout/ensemble/sex/male_confusion_matrix.csv', sep = ",", index = True)







####################### Results #######################
# Ensemble all
pd.read_csv("performance/holdout/ensemble/confusion_matrix.csv")
pd.read_csv("performance/holdout/ensemble/classification_report.csv")
# Ensemble female
pd.read_csv("performance/holdout/ensemble/sex/female_confusion_matrix.csv")
pd.read_csv("performance/holdout/ensemble/sex/female_classification_report.csv")
# Ensemble male
pd.read_csv("performance/holdout/ensemble/sex/male_confusion_matrix.csv")
pd.read_csv("performance/holdout/ensemble/sex/male_classification_report.csv")

# Submodels
# Coefs
pd.read_csv("performance/holdout/coefs_all_models.csv")
# 1
pd.read_csv("performance/holdout/models/classification_report1.csv")
pd.read_csv("performance/holdout/models/confusion_matrix1.csv")
# 2
pd.read_csv("performance/holdout/models/classification_report2.csv")
pd.read_csv("performance/holdout/models/confusion_matrix2.csv")
# 3
pd.read_csv("performance/holdout/models/classification_report3.csv")
pd.read_csv("performance/holdout/models/confusion_matrix3.csv")
# 4
pd.read_csv("performance/holdout/models/classification_report4.csv")
pd.read_csv("performance/holdout/models/confusion_matrix4.csv")
# 5
pd.read_csv("performance/holdout/models/classification_report5.csv")
pd.read_csv("performance/holdout/models/confusion_matrix5.csv")