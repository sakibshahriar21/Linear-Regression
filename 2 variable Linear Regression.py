import pandas as pd
#Step 1: Readding dataset

df = pd.read_csv('Dataset for Multi.csv') #take any dataset with .csv format

#Step 2: Parameter Initilization

theta = [1,2,1.5]

x1 = df['x1']
x2 = df['x2']
y = df["y"]
#print(x1,x2,y)

count = 0
for i in range(10):

    print()
    print("iteration number:", count)
    count += 1
    h = []
    m = len(x1)

    # Step 3: Hypothesis Function

    for i in range(m):
        h.append(theta[0] + theta[1] * x1[i] + theta[2] * x2[i])


    # Step 4: cost function
    error = 0

    for i in range(m):
        error = error + (h[i] - y[i]) ** 2

    J = (1 / (2 * m)) * error
    print("Cost= ",J)

    # Step 5: Gradinet Descent

    alpha = 0.01
    diff_J_theta0 = 0
    diff_J_theta1 = 0
    diff_J_theta2 = 0

    for i in range(m):

        diff_J_theta0 = diff_J_theta0 + (h[i] - y[i])
        diff_J_theta1 = diff_J_theta1 + (h[i] - y[i]) * x1[i]
        diff_J_theta2 = diff_J_theta2 + (h[i] - y[i]) * x2[i]

    theta[0] = theta[0] - (alpha / m) * diff_J_theta0

    theta[1] = theta[1] - (alpha / m) * diff_J_theta1

    theta[2] = theta[2] - (alpha / m) * diff_J_theta2

    print("Theta 0= ",theta[0])
    print("Theta 1= ",theta[1])
    print("Theta 2= ",theta[2])


