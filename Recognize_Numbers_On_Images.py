from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
#print(digits.DESCR)
#a = digits.target.shape
#print(a)
#b = digits.data.shape
#print(b)
#c = digits.images.shape
#print(c)

x = digits.images[0]
plt.gray()
plt.imshow(x)
#plt.show()

