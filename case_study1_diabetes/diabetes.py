""" 
Summary of the program:
- import lib's
- read dataset
- data.describe()
- replace NAN values with mean
- Handle outliers
- divide into independent and dependent features
- perform standardscaler scaling
- check for multicollinearity using VIF 
- split into train and test set
- train using gaussian NB
- check performance using accuracy score, confusion matrix, TP, FP, FN, TN, Precision, recal, f1 score, AUC, plot AUC
- train on logistic regression and naive bayes
- check performance metric using accuracy, confusion matrix and ROC """

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skl
sns.set()

# Reading the dataset
data = pd.read_csv("case_study1_diabetes/diabetes.csv") 
data.head()
""" 
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1 """

data.describe()
""" 
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000 

- we can see there few data for columns Glucose, Insulin, skin thickness, BMI and Blood Pressure which have value as 0. 
- That's not possible. You can do a quick search to see that one cannot have 0 values for these. 
- Let's deal with that. we can either remove such data or simply replace it with their respective mean values. 
- Let's do it later.
"""

# replacing zero values with the mean of the column
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())

# Handling the Outliers
q = data['Pregnancies'].quantile(0.98)
# 12.0

# we are removing the top 2% data from the Pregnancies column
data_cleaned = data[data['Pregnancies']<q]
q = data_cleaned['BMI'].quantile(0.99)
# 51.28799999999987

# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)
# 50.0

# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)

# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
# 293.0

# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
q = data_cleaned['Age'].quantile(0.99)
# 67.0

# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data_cleaned:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data_cleaned[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()

# divide into independent and dependent features
X = data.drop(columns = ['Outcome'])
y = data['Outcome']

# we need to scale our data as well
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# This is how our data looks now after scaling.
X_scaled
""" array([[ 0.63994726,  0.86527574, -0.0210444 , ...,  0.16725546,
         0.46849198,  1.4259954 ],
       [-0.84488505, -1.20598931, -0.51658286, ..., -0.85153454,
        -0.36506078, -0.19067191],
       [ 1.23388019,  2.01597855, -0.68176235, ..., -1.33182125,
         0.60439732, -0.10558415],
       ...,
       [ 0.3429808 , -0.02240928, -0.0210444 , ..., -0.90975111,
        -0.68519336, -0.27575966],
       [-0.84488505,  0.14197684, -1.01212132, ..., -0.34213954,
        -0.37110101,  1.17073215],
       [-0.84488505, -0.94297153, -0.18622389, ..., -0.29847711,
        -0.47378505, -0.87137393]]) """

# now we will check for multicollinearity using VIF(Variance Inflation factor)
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif
""" 
        vif                  Features
0  1.431075               Pregnancies
1  1.347308                   Glucose
2  1.247914             BloodPressure
3  1.450510             SkinThickness
4  1.262111                   Insulin
5  1.550227                       BMI
6  1.058104  DiabetesPedigreeFunction
7  1.605441                       Age 

- All the VIF values are less than 5 and are very low. 
- That means no multicollinearity. 
- Now, we can go ahead with fitting our data to the model. 

Before that, let's split our data in test and training set."""


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25, random_state = 355)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train,y_train)
# GaussianNB()

import pickle

# Writing different model files to file
with open( 'case_study1_diabetes/modelForPrediction.sav', 'wb') as f:
    pickle.dump(model,f)
    
with open('case_study1_diabetes/standardScalar.sav', 'wb') as f:
    pickle.dump(scalar,f)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
# 0.7864583333333334

# Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat
""" array([[109,  16],
       [ 25,  42]], dtype=int64) """

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy
# 0.7864583333333334

# Precison
Precision = true_positive/(true_positive+false_positive)
Precision
# 0.872

# Recall
Recall = true_positive/(true_positive+false_negative)
Recall

# 0.8134328358208955

# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score
# 0.8416988416988417

# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
auc
# 0.7494328358208956

""" 
- So far we have been doing grid search to maximise the accuracy of our model. 
- Here, we’ll follow a different approach. 
- We’ll create two models, one with Logistic regression and other with Naïve Bayes and we’ll compare the AUC. 
- The algorithm having a better AUC shall be considered for production deployment. """

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayes')
plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)
# LogisticRegression()

y_pred_logistic = log_reg.predict(x_test)

accuracy_logistic = accuracy_score(y_test,y_pred_logistic)
accuracy_logistic
# 0.7552083333333334

# Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred_logistic)
conf_mat
""" array([[110,  15],
            [ 32,  35]], dtype=int64) """

# ROC
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_test, y_pred_logistic)
plt.plot(fpr_logistic, tpr_logistic, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
plt.legend()
plt.show()

from sklearn.metrics  import roc_auc_score

auc_naive=roc_auc_score(y_test,y_pred)
auc_naive
# 0.7494328358208956

auc_logistic=roc_auc_score(y_test,y_pred_logistic)
auc_logistic
# 0.7011940298507463