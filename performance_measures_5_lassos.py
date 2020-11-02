# Importing libraries and functions we'll use
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import pandas as pd
import numpy as np

# Importing all the folds
train1 = [pd.read_csv('./csv_files/train1.csv'),1]
train2 = [pd.read_csv('./csv_files/train2.csv'),2]
train3 = [pd.read_csv('./csv_files/train3.csv'),3]
train4 = [pd.read_csv('./csv_files/train4.csv'),4]
train5 = [pd.read_csv('./csv_files/train5.csv'),5]

################## Creating a loop, so I won't have to do the below code 5 times ##################
# Making the datasets into a list
datasets = [train1, train2, train3, train4, train5]
classif_reports = ["", "", "", "", ""]
conf_mtxs = []
model = SVC(kernel = 'linear')
SVM_coef = pd.DataFrame(columns = ['predictor_name', 'coef', 'fold'])

for df, n in datasets:
    # Divide the data up into train and test
    df_test = df.loc[df['fold'] == n]
    df_train = df.loc[df['fold'] != n]
    
    # Dividing train and test up into predictor variables, and Diagnosis
    x_train1 = df_train.iloc[:,5:]
    y_train1 = df_train.loc[:,['ID', 'Diagnosis']]
    y_train1 = y_train1.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders
    
    x_test1 = df_test.iloc[:,5:]
    y_test1 = df_test.loc[:,['ID', 'Diagnosis']]
    y_test1 = y_test1.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    # Fit the data onto the model
    model.fit(x_train1, y_train1)

    # Predict the test set (on the basis of predictor variables), using the fitted model
    predictions_test1 = model.predict(x_test1)

    # Get out the coefficients
    coef1 = model.coef_
    coef1 = coef1[0]

    # Get out the coefficient names
    coef1_names = list(x_test1)
    coef1_names = np.asarray(coef1_names)

    # Create an array with info on which folds data/features we're handling!
    fold = np.repeat(n, len(coef1))

    # Load in the predictor names, and their coefficients to the empty dataframe
    SVM_coef = SVM_coef.append(pd.DataFrame({'predictor_name': coef1_names, 'coef' : coef1, 'fold' : fold}), ignore_index = True)

    # Get classification report
    report = classification_report(y_test1, predictions_test1, output_dict = True)
    report_number = n-1
    classif_reports[report_number] = report
    
    # Get confusion matrix
    matrixx = confusion_matrix(y_test1, predictions_test1)
    conf_mtxs.append(matrixx)

    # The end


# Loading each element (classification_report) of the list of dictionaries, as a dataframe 
classif_report_fold1 = pd.DataFrame(classif_reports[0])
classif_report_fold2 = pd.DataFrame(classif_reports[1])
classif_report_fold3 = pd.DataFrame(classif_reports[2])
classif_report_fold4 = pd.DataFrame(classif_reports[3])
classif_report_fold5 = pd.DataFrame(classif_reports[4])

# Loading each element (classification_report) of the list of dictionaries, as a dataframe
conf_matrix_fold1 = pd.DataFrame(conf_mtxs[0])
conf_matrix_fold2 = pd.DataFrame(conf_mtxs[1])
conf_matrix_fold3 = pd.DataFrame(conf_mtxs[2])
conf_matrix_fold4 = pd.DataFrame(conf_mtxs[3])
conf_matrix_fold5 = pd.DataFrame(conf_mtxs[4])

# Give the matrices names for rows and columns
conf_matrix_fold1.columns = conf_matrix_fold1.columns = ['predict_sz', 'predict_td']
conf_matrix_fold1.index = conf_matrix_fold1.index = ['actual_sz', 'actual_td']
conf_matrix_fold2.columns = conf_matrix_fold2.columns = ['predict_sz', 'predict_td']
conf_matrix_fold2.index = conf_matrix_fold2.index = ['actual_sz', 'actual_td']
conf_matrix_fold3.columns = conf_matrix_fold3.columns = ['predict_sz', 'predict_td']
conf_matrix_fold3.index = conf_matrix_fold3.index = ['actual_sz', 'actual_td']
conf_matrix_fold4.columns = conf_matrix_fold4.columns = ['predict_sz', 'predict_td']
conf_matrix_fold4.index = conf_matrix_fold4.index = ['actual_sz', 'actual_td']
conf_matrix_fold5.columns = conf_matrix_fold5.columns = ['predict_sz', 'predict_td']
conf_matrix_fold5.index = conf_matrix_fold5.index = ['actual_sz', 'actual_td']

# Writing everything to .csv
SVM_coef.to_csv('./performance_measures_5_lassos/SVM_coef.csv', sep=',', index = True)

# Writing confusion matrices to csv
conf_matrix_fold1.to_csv('./performance_measures_5_lassos/conf_matrix_fold1.csv', sep=',', index = True)
conf_matrix_fold2.to_csv('./performance_measures_5_lassos/conf_matrix_fold2.csv', sep=',', index = True)
conf_matrix_fold3.to_csv('./performance_measures_5_lassos/conf_matrix_fold3.csv', sep=',', index = True)
conf_matrix_fold4.to_csv('./performance_measures_5_lassos/conf_matrix_fold4.csv', sep=',', index = True)
conf_matrix_fold5.to_csv('./performance_measures_5_lassos/conf_matrix_fold5.csv', sep=',', index = True)

# Writing classification reports to csv
classif_report_fold1.to_csv('./performance_measures_5_lassos/classif_report_fold1.csv', sep=',', index = True)
classif_report_fold2.to_csv('./performance_measures_5_lassos/classif_report_fold2.csv', sep=',', index = True)
classif_report_fold3.to_csv('./performance_measures_5_lassos/classif_report_fold3.csv', sep=',', index = True)
classif_report_fold4.to_csv('./performance_measures_5_lassos/classif_report_fold4.csv', sep=',', index = True)
classif_report_fold5.to_csv('./performance_measures_5_lassos/classif_report_fold5.csv', sep=',', index = True)

















