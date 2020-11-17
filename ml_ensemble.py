import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt
import pandas as pd
import numpy as np
import os

os.getcwd()

# Load train and test data
data1 = [pd.read_csv('./csv_files/train1.csv', index_col=0), 1, pd.read_csv('./csv_files/holdout1.csv', index_col=0)]
data2 = [pd.read_csv('./csv_files/train2.csv', index_col=0), 2, pd.read_csv('./csv_files/holdout2.csv', index_col=0)]
data3 = [pd.read_csv('./csv_files/train3.csv', index_col=0), 3, pd.read_csv('./csv_files/holdout3.csv', index_col=0)]
data4 = [pd.read_csv('./csv_files/train4.csv', index_col=0), 4, pd.read_csv('./csv_files/holdout4.csv', index_col=0)]
data5 = [pd.read_csv('./csv_files/train5.csv', index_col=0), 5, pd.read_csv('./csv_files/holdout5.csv', index_col=0)]

# Take the lists and include them in a list
datasets = [data1, data2, data3, data4, data5]

datasets[0][2]

# Defining the model we're using
model = SVC(kernel = 'linear')

# Preparing a dataframe for the predictions
# predictions = pd.DataFrame(columns = ['ID', "diagnosis_real"]) 

predictions = pd.DataFrame({'ID' : datasets[0][2].loc[:,'ID'], 'diagnosis_real' : datasets[0][2].loc[:,'Diagnosis']})

# Checking the datasets
datasets[0][0].iloc[:,0:10]
datasets[0][2].iloc[:,0:10]

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
    
    # ABOVE WORKS, THIS COULD BE ADDED TO LOOP:
    # Make a dataframe with ID and Real Diagnosis
    # predictions = predictions.append(pd.DataFrame({'ID' : holdout.loc[:,'ID'], 'diagnosis_real' : holdout.loc[:,'Diagnosis']}), ignore_index = True)
    # predictions = pd.DataFrame({'ID' : holdout.loc[:,'ID'], 'diagnosis_real' : holdout.loc[:,'Diagnosis']})
    print(len(predictions['ID']))

    # Make a unique name for each iteration
    new_col_name = "".join(["diagnosis_predic_", str(k)])
    
    # Add a column to the dataframe (with unique name, and the predictions)
    predictions[new_col_name] = y_predicted

# Make sure diagnosis_real column also has 1's and 0's instead of 'td' and 'sz'
predictions['diagnosis_real'] = predictions['diagnosis_real'].apply(lambda x: 0 if x=='td' else 1)

# For each row, get a count of 1's and 0's in the new columns
count_of_1_predictions = predictions.iloc[:,-5:].apply(pd.Series.value_counts, axis=1)[1].fillna(0)
predictions['diagnosis_predic_ensemble'] = [0 if x < 3 else 1 for x in count_of_1_predictions]

predictions





















################## Creating a loop, so I won't have to do the below code 5 times ##################

#creating empty objects to save data in
classif_reports = ["", "", "", "", ""]
conf_mtxs = []
model = SVC(kernel = 'linear')
SVM_coef = pd.DataFrame(columns = ['predictor_name', 'coef', 'fold'])

#defining which feature set is currently run 
feature_set = 'performance_measures_5_feature_sets'

for df, n in datasets:
    # making td into 0 and sz into 1
    df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 0 if x=='td' else 1) # setting TD as 0 and ASD as 1
   
    # Divide the data up into train and test
    df_test = df.loc[df['fold'] == n]
    df_train = df.loc[df['fold'] != n]

    # Dividing train and test up into predictor variables, and Diagnosis
    x_train1 = df_train.iloc[:,4:]
    y_train1 = df_train.loc[:,['ID', 'Diagnosis']]
    y_train1 = y_train1.set_index('ID') # Setting index as ID, to be able to map how well the model predicts on genders

    x_test1 = df_test.iloc[:,4:]
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

    #Saving data 
   
    #making variables to use in filenames
    svm_coef_name = "".join(['./performance_measures_5_lassos/', str(feature_set), "_CoefTestFold", ".csv"]) 
    conf_matrix_name = "".join(['./performance_measures_5_lassos/', str(feature_set),"_ConfusionMatrix", str(n), ".csv"]) 
    classification_report_name = "".join(['./performance_measures_5_lassos/', str(feature_set),"_ClassificationReport", str(n), ".csv"]) 


    # Loading each element (classification_report) of the list of dictionaries, as a dataframe 
    classif_report_fold = pd.DataFrame(classif_reports[n-1])
    # Loading each element (classification_report) of the list of dictionaries, as a dataframe
    conf_matrix_fold = pd.DataFrame(conf_mtxs[n-1])
    #Give the matrices names for rows and columns
    conf_matrix_fold.columns = conf_matrix_fold.columns = ['predict_asd', 'predict_td']
    conf_matrix_fold.index = conf_matrix_fold.index = ['actual_asd', 'actual_td']

    # Writing coefficients to .csv
    SVM_coef.to_csv(svm_coef_name, sep=',', index = True)
    # Writing confusion matrices to csv
    conf_matrix_fold.to_csv(conf_matrix_name, sep=',', index = True)
    # Writing classification reports to csv
    classif_report_fold.to_csv(classification_report_name, sep=',', index = True)


    # The end

