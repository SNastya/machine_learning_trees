
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 


df = pd.read_csv('C:/Users/Анастасия/Desktop/прога/екатерина/vehicles_dataset_old.csv')

df_prepared = df.copy()
df_prepared = df_prepared.drop(['price'], axis=1)


y = df_prepared['price_category']

df_try = df_prepared.iloc[3]

try1 = df_try.values.tolist()
try2 = df_try.index.to_list()

a = []

for i in range(len(try1)):
    try:
        float(try1[i])
    except:
        a.append(try2[i])

df_prepared = df_prepared.drop(a, axis = 1)

x = df_prepared


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)

RandomForestClassifier(n_estimators=10000, max_features='sqrt',  max_depth=4)


predicted_test_rf = rf_clf.predict(x_test)
print(accuracy_score(y_test, predicted_test_rf))
print(confusion_matrix(y_test, predicted_test_rf))