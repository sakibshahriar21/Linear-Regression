import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings

#Step 1: Reading dataset

df = pd.read_csv('cholesterol_data.csv')

encoder = LabelEncoder()


x1 = df['x1'] #x1 column represents height(inches)
x2 = df['x2'] #x2 column represents Weight(kilograms)
x3 = df['x3'] #x3 column represents Age(Years)
y = df["y"] #y coulnmn represents cholesterol

df['x1'] = encoder.fit_transform(df['x1'])
df['x2'] = encoder.fit_transform(df['x2'])
df['x3'] = encoder.fit_transform(df['x3'])
df['y'] = encoder.fit_transform(df['y'])

#Step 2: Parameter initialization
theta=[df['x1'].mean(),df['x2'].mean(),df['x3'].mean(),df['y'].mean()]

count = 0
for i in range(10):

    print()
    print("iteration number:", count)
    count += 1
    h = [] #hypothesis list
    m = len(x1)

    # Step 3: Hypothesis Function
    for i in range(m):
        h.append(theta[0] + theta[1] * x1[i] + theta[2] * x2[i] + theta[3] * x3[i])


    # Step 4: cost function
    error = 0

    for i in range(m):
        error = error + (h[i] - y[i]) ** 2

    J = (1 / (2 * m)) * error
    print("Cost= ",J)

    # Step 5: Gradinet Descent

    alpha = 0.001
    diff_J_theta0 = 0
    diff_J_theta1 = 0
    diff_J_theta2 = 0
    diff_J_theta3 = 0

    for i in range(m):

        diff_J_theta0 = diff_J_theta0 + (h[i] - y[i])
        diff_J_theta1 = diff_J_theta1 + (h[i] - y[i]) * x1[i]
        diff_J_theta2 = diff_J_theta2 + (h[i] - y[i]) * x2[i]
        diff_J_theta3 = diff_J_theta3 + (h[i] - y[i]) * x3[i]

    theta[0] = theta[0] - (alpha / m) * diff_J_theta0

    theta[1] = theta[1] - (alpha / m) * diff_J_theta1

    theta[2] = theta[2] - (alpha / m) * diff_J_theta2

    theta[3] = theta[3] - (alpha / m) * diff_J_theta3

    for i in range(4):
        print('Theta',i,'= ',theta[i])

    #X = df.x2
    #V = df.x3
    #Y = df.y
    #warnings.simplefilter(action="ignore", category=FutureWarning)
    #sns.regplot(X, Y)
    #sns.regplot(V, Y)
    #sns.scatterplot(df.x2, df.y)
    #sns.scatterplot(df.x3, df.y)


