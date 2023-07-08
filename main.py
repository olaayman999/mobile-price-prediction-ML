import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# calculates and visualizes a confusion matrix based on the predicted and actual target values. It also displays the classification report and returns the confusion matrix.
#(actual target values from the test set, predicted target values obtained from the model, title for the confusion matrix plot)
def confusionMX(y_test, y_pred, plt_title):
    CMX=confusion_matrix(y_test, y_pred) #compares the actual target values with the predicted target values
    print(classification_report(y_test, y_pred))
    sns.heatmap(CMX, annot=True, fmt='g', cbar=False, cmap='BuPu')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return CMX

trainData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
#displays the first few rows of the loaded data
# print(trainData.head()) 
# print(testData.head())
# #obtain a concise summary of the DataFrame train_data, providing information about the dataset's structure, column data types, and missing values.
# print(trainData.info())  

#checks if the values in the 'screen width' column of trainData are not equal to zero.
#applies the Boolean mask to the trainData, filtering out rows where 'sc_w' is zero.
trainDataFiltered = trainData[trainData['sc_w'] != 0]
trainDataFiltered = trainDataFiltered [trainDataFiltered['sc_h'] != 0]
trainDataFiltered = trainDataFiltered [trainDataFiltered['mobile_wt' ] != 0]
trainDataFiltered = trainDataFiltered [trainDataFiltered['m_dep'] != 0 ]
trainDataFiltered = trainDataFiltered [trainDataFiltered['px_width'] != 0 ]
trainDataFiltered = trainDataFiltered [trainDataFiltered['px_height'] != 0 ]


testDataFiltered = testData[testData['sc_w'] != 0]
testDataFiltered =testDataFiltered[testDataFiltered['sc_h'] != 0 ]
testDataFiltered =testDataFiltered[testDataFiltered['mobile_wt' ] != 0  ]
testDataFiltered =testDataFiltered[testDataFiltered['m_dep'] != 0 ]
testDataFiltered =testDataFiltered[testDataFiltered['px_width'] != 0 ]
testDataFiltered =testDataFiltered[testDataFiltered['px_height'] != 0  ]

# output the dimensions of the DataFrame, such as (rows, columns)
# print(f'dataframe dimentions before filtering {trainData.shape}')
# print(f'dataframe dimentions after filtering {trainDataFiltered.shape}')


def snsGraphstrain(flag, y_pred_svm=0):
    if flag==0:
        sns.set()
        price_plot=trainDataFiltered['price_range'].value_counts().plot(kind='bar')
        plt.xlabel('price_range')
        plt.ylabel('Count')
        plt.show()

        sns.set()
        ax=sns.displot(data=trainDataFiltered["battery_power"])
        plt.show()

        sns.set()
        ax=sns.displot(data=trainDataFiltered["blue"])
        plt.xlabel('bluetooth')
        plt.show()

    else:
        y_pred_svm_series = pd.Series(y_pred_svm) #convert it to a pandas Series before calling the value_counts()
        sns.set()
        price_plot = y_pred_svm_series.value_counts().plot(kind='bar')
        plt.xlabel('price_range')
        plt.ylabel('Count')
        plt.show()

snsGraphstrain(0)


X=trainDataFiltered.drop(['price_range'], axis=1) #input feature matrix
y=trainDataFiltered['price_range'] # creates the target variable array
Xtst=testDataFiltered.drop(['id'], axis=1)

#train_test_split(Input feature matrix, target variable array, proportion of the data to be allocated to the validation set, random seed for reproducibility.) 
# 20% of the data will be used for validation.
X_train, X_valid, y_train, y_valid= train_test_split(X, y, test_size=0.2, random_state=7)
#  The training set contains 80% of the original data, and the validation set contains 20% of the original data.


#-------------------------Random Forest Classifier--------------------------------
def RFV():
    from sklearn.ensemble import RandomForestClassifier

    rfc=RandomForestClassifier()
    rfc.fit(X_train, y_train) #train
    y_pred_rfc=rfc.predict(X_valid) 

    print('Random Forest Classifier Accuracy Score: ',accuracy_score(y_valid,y_pred_rfc))
    cm_rfc=confusionMX(y_valid, y_pred_rfc, 'Random Forest Confusion Matrix')


#----------------------------Gaussian NB classifier--------------------------
def gnb():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()

    gnb.fit(X_train, y_train)
    y_pred_gnb=gnb.predict(X_valid)

    print('Gaussian NB Classifier Accuracy Score: ',accuracy_score(y_valid,y_pred_gnb))
    cm_rfc=confusionMX(y_valid, y_pred_gnb, 'Gaussian NB Confusion Matrix')

#----------------------------------KNN Classifier------------------------------------
def knn():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3,leaf_size=25)

    knn.fit(X_train, y_train)
    y_pred_knn=knn.predict(X_valid)
    print('KNN Classifier Accuracy Score: ',accuracy_score(y_valid,y_pred_knn))
    cm_rfc=confusionMX(y_valid, y_pred_knn, 'KNN Confusion Matrix')

#-----------------------------------------SVM Classifier----------------------------------
def SVM(flag):
    from sklearn import svm
    svm_clf = svm.SVC(decision_function_shape='ovo')

    svm_clf.fit(X_train, y_train)
    if flag==0: #train
        y_pred_svm=svm_clf.predict(X_valid)
        print('SVM Classifier Accuracy Score: ',accuracy_score(y_valid,y_pred_svm))
        cm_rfc=confusionMX(y_valid, y_pred_svm, 'SVM Confusion Matrix')
    else:
        y_pred_svm=svm_clf.predict(Xtst)
        for i in range(len(y_pred_svm)):
            print(y_pred_svm[i])
        snsGraphstrain(1, y_pred_svm)


RFV()
knn()
gnb()
SVM(0)
SVM(1)

