import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_linear_regression
import numpy as np
import seaborn as sns
#Step 1: Readding dataset

df = pd.read_csv('dataset.csv')
#sns.set_theme(color_codes=True)


#Step 2: Parameter Initilization

theta = [1,2]

x = df['x']
y = df['y']
#print(x,y)

count = 0
for i in range(10):

    print()
    print("iteration number:", count)
    count = count + 1
    h = []
    m = len(x)

    #Step 3: Hypothesis Function

    for i in range(m):
        h.append(theta[0] + theta[1] * x[i])

    # print(h[3])

    #Step 4: cost function
    error = 0

    for i in range(m):
        error = error + (h[i] - y[i]) ** 2

    J = (1 / (2 * m)) * error

    print('Cost= ',J)

    #Step 5: Gradinet Descent

    alpha = 0.01
    diff_J_theta0 = 0
    diff_J_theta1 = 0

    for i in range(m):

        diff_J_theta0 = diff_J_theta0 + (h[i] - y[i])
        diff_J_theta1 = diff_J_theta1 + (h[i] - y[i]) * x[i]

    theta[0] = theta[0] - (alpha / m) * diff_J_theta0

    theta[1] = theta[1] - (alpha / m) * diff_J_theta1

    print("Theta 0 =",theta[0]," Theta 1 =",theta[1])

    #X = np.array(theta[0])
    #y = np.array(theta[1])
    #intercept, slope, corr_coeff = plot_linear_regression(X, y)
    #plt.show()



