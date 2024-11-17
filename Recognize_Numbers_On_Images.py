from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
#print(digits.DESCR)
#a = digits.target
#print(a)
#b = digits.data.shape
#print(b)
#c = digits.images.shape
#print(c)
x = digits.images[0]
print(digits.target[0])
plt.gray()
plt.imshow(x)
plt.show()

x_train,x_test,y_train,y_test = train_test_split(digits.data, digits.target, test_size=30)
scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def calculat_metrics(y_train,y_test,y_pred_train,y_pred_test):
    accu_train = accuracy_score(y_true= y_train, y_pred= y_pred_train)
    accu_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)

    p = precision_score(y_true=y_test, y_pred=y_pred_test, average= "weighted")
    r = recall_score(y_true=y_test, y_pred=y_pred_test, average= "weighted")

    print(f"acc_train: {accu_train} - acc_test: {accu_test} - precision: {p} - recall: {r} ")
    return accu_train,accu_test,p,r


rf = RandomForestClassifier(n_estimators= 100, max_depth= 101)
rf.fit(x_train,y_train)
y_pred_train = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

acc_train_rf,acc_test_rf,p_rf,r_rf = calculat_metrics(y_train,y_test,y_pred_train,y_pred_test)