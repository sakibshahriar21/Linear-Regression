import pandas as pd
from sklearn.preprocessing import LabelEncoder
#Step 1: Readding dataset

df = pd.read_csv('Real estate.csv')
#scaler = StandardScaler()

#scaled_features = StandardScaler().fit_transform(df.values)
#scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

#Step 2: Parameter Initilization

#theta = [3.5,4.75,6.25,5.75,8.5,5,12.5]

#theta = [1998.7,1088,550.4,0.1,41.1,70.1,288.9]

encoder = LabelEncoder()

x1 = df['x1']
x2 = df['x2']
x3 = df['x3']
x4 = df['x4']
x5 = df['x5']
x6 = df['x6']
y = df["y"]

df['x1'] = encoder.fit_transform(df['x1'])
df['x2'] = encoder.fit_transform(df['x2'])
df['x3'] = encoder.fit_transform(df['x3'])
df['x4'] = encoder.fit_transform(df['x4'])
df['x5'] = encoder.fit_transform(df['x5'])
df['x6'] = encoder.fit_transform(df['x6'])
df['y'] = encoder.fit_transform(df['y'])

theta=[df['x1'].mean(),df['x2'].mean(),df['x3'].mean(),df['x4'].mean(),df['x5'].mean(),
        df['x6'].mean(),df['y'].mean()]
#print(x1,x2,y)
#x1,x2,x3,x4,x5,x6,y=(theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7])


count = 0
for i in range(1000):

    print()
    print("iteration number:", count)
    count += 1
    h = [] #hypothesis
    m = len(x1)

    # Step 3: Hypothesis Function

    for i in range(m):
        h.append(theta[0] + theta[1] * x1[i] + theta[2] * x2[i] + theta[3] * x3[i] +
                 theta[4] * x4[i] + theta[5] * x5[i] + theta[6] * x6[i])

    #print(h[2])

    # Step 4: cost function
    error = 0

    for i in range(m):
        error = error + (h[i] - y[i]) ** 2

    J = (1 / (2 * m)) * error
    print("Cost= ",J)

    # Step 5: Gradinet Descent

    alpha = 0.00001
    diff_J_theta0 = 0
    diff_J_theta1 = 0
    diff_J_theta2 = 0
    diff_J_theta3 = 0
    diff_J_theta4 = 0
    diff_J_theta5 = 0
    diff_J_theta6 = 0

    for i in range(m):

        diff_J_theta0 = diff_J_theta0 + (h[i] - y[i])
        diff_J_theta1 = diff_J_theta1 + (h[i] - y[i]) * x1[i]
        diff_J_theta2 = diff_J_theta2 + (h[i] - y[i]) * x2[i]
        diff_J_theta3 = diff_J_theta3 + (h[i] - y[i]) * x3[i]
        diff_J_theta4 = diff_J_theta4 + (h[i] - y[i]) * x4[i]
        diff_J_theta5 = diff_J_theta5 + (h[i] - y[i]) * x5[i]
        diff_J_theta6 = diff_J_theta6 + (h[i] - y[i]) * x6[i]

    theta[0] = theta[0] - (alpha / m) * diff_J_theta0

    theta[1] = theta[1] - (alpha / m) * diff_J_theta1

    theta[2] = theta[2] - (alpha / m) * diff_J_theta2

    theta[3] = theta[3] - (alpha / m) * diff_J_theta3

    theta[4] = theta[4] - (alpha / m) * diff_J_theta4

    theta[5] = theta[5] - (alpha / m) * diff_J_theta5

    theta[6] = theta[6] - (alpha / m) * diff_J_theta6

    for j in range(7):
        print('Theta',j,'= ',theta[j])


