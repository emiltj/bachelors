import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import pandas as pd
import numpy as np

data1 = [pd.read_csv('./csv_files/train1.csv', index_col=0), 1, pd.read_csv('./csv_files/test1.csv')]
data2 = [pd.read_csv('./csv_files/train2.csv', index_col=0), 2, pd.read_csv('./csv_files/test2.csv')]
data3 = [pd.read_csv('./csv_files/train3.csv', index_col=0), 3, pd.read_csv('./csv_files/test3.csv')]
data4 = [pd.read_csv('./csv_files/train4.csv', index_col=0), 4, pd.read_csv('./csv_files/test4.csv')]
data5 = [pd.read_csv('./csv_files/train5.csv', index_col=0), 5, pd.read_csv('./csv_files/test5.csv')]

################## Creating a loop, so I won't have to do the below code 5 times ##################
# Making the datasets into a list
datasets = [data1, data2, data3, data4, data5]

#creating empty objects to save data in
classif_reports = ["", "", "", "", ""]
conf_mtxs = []
model = SVC(kernel = 'linear', class_weight = 'balanced')
SVM_coef = pd.DataFrame(columns = ['predictor_name', 'coef', 'fold'])

#defining which feature set is currently run 
feature_set = 'test_performance'

for train, n, test in datasets:   
    # Dividing train and test up into predictor variables, and Diagnosis
    x = train.iloc[:,4:]
    y = train.loc[:,['ID', 'Diagnosis']]
    y = y.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    x_test = test.iloc[:,5:]
    y_test = test.loc[:,['ID', 'Diagnosis']]
    y_test = y_test.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    # Fit the data onto the model
    model.fit(x, y)

    # Predict the test set (on the basis of predictor variables), using the fitted model
    y_predictions = model.predict(x_test)

    print(y_predictions)

    # Get out the coefficients
    coefs = model.coef_[0]
    
    # Get out the coefficient names
    coefs_names = list(x_test)
    coefs_names = np.asarray(coefs_names)

    # Create an array with info on which folds data/features we're handling!
    fold = np.repeat(n, len(coefs))

    # Load in the predictor names, and their coefficients to the empty dataframe
    SVM_coef = SVM_coef.append(pd.DataFrame({'predictor_name': coefs_names, 'coef' : coefs, 'fold' : fold}), ignore_index = True)

    # Get classification report
    report = classification_report(y_test, y_predictions, output_dict = True)
    report_number = n-1
    classif_reports[report_number] = report
    
    # Get confusion matrix
    matrixx = confusion_matrix(y_test, y_predictions)
    conf_mtxs.append(matrixx)
    classif_reports[4]

    #Saving data
   
    # making variables to use in filenames
    svm_coef_name = "".join(['./performance/test/', str(feature_set), "_coef_test_fold", ".csv"]) 
    conf_matrix_name = "".join(['./performance/test/', str(feature_set),"_confusion_matrix", str(n), ".csv"]) 
    classification_report_name = "".join(['./performance/test/', str(feature_set),"_classification_report", str(n), ".csv"]) 


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

pd.read_csv("performance/test/test_performance_classification_report1.csv")
pd.read_csv("performance/test/test_performance_classification_report2.csv")
pd.read_csv("performance/test/test_performance_classification_report3.csv")
pd.read_csv("performance/test/test_performance_classification_report4.csv")
pd.read_csv("performance/test/test_performance_classification_report5.csv")

pd.read_csv("performance/test/test_performance_confusion_matrix1.csv")
pd.read_csv("performance/test/test_performance_confusion_matrix2.csv")
pd.read_csv("performance/test/test_performance_confusion_matrix3.csv")
pd.read_csv("performance/test/test_performance_confusion_matrix4.csv")
pd.read_csv("performance/test/test_performance_confusion_matrix5.csv")

pd.read_csv("performance/test/test_performance_coef_test_fold.csv")

