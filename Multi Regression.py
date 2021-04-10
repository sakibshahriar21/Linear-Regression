import pandas as pd
#Step 1: Readding dataset

df = pd.read_csv('patient.csv')

#Step 2: Parameter Initilization


x1 = df['x1']
x2 = df['x2']
Y = df['Y']
#print(x1,x2,y)

theta=[df['x1'].mean(),df['x2'].mean(),df['Y'].mean()]

count = 0
for i in range(1000):

    print()
    print("iteration number:", count)
    count += 1
    h = [] #hypothesis
    m = len(x1)

    # Step 3: Hypothesis Function

    for i in range(m):
        h.append(theta[0] + theta[1] * x1[i] + theta[2] * x2[i])

    #print(h[2])

    # Step 4: cost function
    error = 0

    for i in range(m):
        error = error + (h[i] - Y[i]) ** 2

    J = (1 / (2 * m)) * error
    print("Cost= ",J)

    # Step 5: Gradinet Descent

    alpha = 0.0001
    diff_J_theta0 = 0
    diff_J_theta1 = 0
    diff_J_theta2 = 0

    for i in range(m):

        diff_J_theta0 = diff_J_theta0 + (h[i] - Y[i])
        diff_J_theta1 = diff_J_theta1 + (h[i] - Y[i]) * x1[i]
        diff_J_theta2 = diff_J_theta2 + (h[i] - Y[i]) * x2[i]

    theta[0] = theta[0] - (alpha / m) * diff_J_theta0

    theta[1] = theta[1] - (alpha / m) * diff_J_theta1

    theta[2] = theta[2] - (alpha / m) * diff_J_theta2

    for i in range(3):
        print('Theta',i,'= ',theta[i])


