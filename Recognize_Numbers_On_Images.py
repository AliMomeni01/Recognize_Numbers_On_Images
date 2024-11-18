from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
#print(digits.DESCR)
#a = digits.target
#print(a)
#b = digits.data.shape
#print(b)
#c = digits.images.shape
#print(c)
x = digits.images[0]
#print(digits.target[0])
plt.gray()
plt.imshow(x)
#plt.show()

x_train,x_test,y_train,y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=45)
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

svm = SVC(kernel= "linear")
svm.fit(x_train,y_train)
y_pred_train = svm.predict(x_train)
y_pred_test = svm.predict(x_test)
acc_train_svm,acc_test_svm,p_svm,r_svm = calculat_metrics(y_train,y_test,y_pred_train,y_pred_test)

ann = MLPClassifier(random_state=45 ,hidden_layer_sizes=256, solver= "lbfgs",max_iter=250, learning_rate_init=0.001)
ann.fit(x_train,y_train)
y_pred_train = ann.predict(x_train)
y_pred_test = ann.predict(x_test)

acc_train_ann,acc_test_ann,p_ann,r_ann = calculat_metrics(y_train,y_test,y_pred_train,y_pred_test)

kn = KNeighborsClassifier(n_neighbors=8)
kn.fit(x_train,y_train)
y_pred_train = kn.predict(x_train)
y_pred_test = kn.predict(x_test)

acc_train_kn,acc_test_kn,p_kn,r_kn = calculat_metrics(y_train,y_test,y_pred_train,y_pred_test)
