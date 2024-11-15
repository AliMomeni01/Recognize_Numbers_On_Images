from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
print (scaler)