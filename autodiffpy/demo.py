import pandas as pd
import autodiff as ad
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("demo.csv")

x = data.drop('y', axis = 1)
y_true = data.y

w = ad.autodiff('w', [1 for i in x.columns.values])


f2 = ad.logistic(w*x)
g = ad.gradient_descent(f2, y_true, loss = 'MSE', beta= 0.001, max_iter = 10000, tol=10**(-8))

print("initial loss:", g["loss_array"][0])
print("final loss:", g["loss_array"][-1])
print(f"final weights: {g['w'].val}")

xgrid = np.linspace(1, g['num_iter'], g["num_iter"])
plt.plot(xgrid, g['loss_array'])
plt.show()
