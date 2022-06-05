import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 

df = pd.read_csv('C:/Users/Анастасия/Desktop/прога/екатерина/vehicles_dataset_prepared.csv')

df_prepared = df.copy()
df_prepared = df_prepared.drop(['price', 'odometer/price_std'], axis=1)

x = df_prepared.drop(['price_category'], axis=1)
y = df_prepared['price_category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
  
DecisionTreeClassifier(random_state=42)

predicted_test = clf.predict(x_test)

accuracy_score(y_test, predicted_test)
confusion_matrix(y_test, predicted_test) 

f_imp_x_train = list(zip(x_train.columns, clf.feature_importances_))
f_imp_x_test = list(zip(x_test.columns, clf.feature_importances_))




def del_Feature(f_imp_list):
    a = []
    f_imp_list.sort(key = lambda x: x[1], reverse = True)
    for i in range(len(f_imp_list)):
        if f_imp_list[i][1] != 0:
            a.append(f_imp_list[i][0])
    return a



x_test = x_test[(del_Feature(f_imp_x_test))]
x_train = x_train[(del_Feature(f_imp_x_train))]

clf.fit(x_train, y_train)  
DecisionTreeClassifier(random_state=42)

predicted_test = clf.predict(x_test)

# print(accuracy_score(y_test, predicted_test))
print(confusion_matrix(y_test, predicted_test))

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)


RandomForestClassifier(n_estimators=10000, max_features='sqrt',  max_depth=4)

predicted_test_rf = rf_clf.predict(x_test)
print(accuracy_score(y_test, predicted_test_rf))
print(confusion_matrix(y_test, predicted_test_rf))

