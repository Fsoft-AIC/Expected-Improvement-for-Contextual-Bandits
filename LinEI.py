import numpy as np;
import numpy.random as ra;
import numpy.linalg as la;
from scipy.stats import norm
from random import randrange


'''created on 26/07/2021'''
'''Declare the variables'''

T = 10000  # number of observations
d = 10   # number of dimensions
K = 5  # number of actions
S = 1.0
R = 1;


'''generate the array of N_a'''
N = [0] * K

lammda = .1
delta = 0.1;
S_hat = 1;
my_c = .1
t = 1;

'''alpha '''
alpha = 1 + np.sqrt(np.log(2/delta)/2)


''' - generate theta_star '''
theta_star = ra.normal(0, 1, d)

'''theta_hat '''
theta_hat = np.zeros(d)


A = np.eye(d)

''' inverse matrix of V_t '''
invA = np.zeros((d,d))


b = np.zeros(d)


def sample_x():
    x = np.random.randn(d)
    normalization = np.linalg.norm(x, 2)
    return x / normalization


x = np.zeros((K, d))


''' Expected Improvement '''
'''theta_hat '''
theta_hat = np.zeros(d)


A = np.eye(d)

''' inverse matrix of V_t '''
invA = np.zeros((d,d))


b = np.zeros(d)

regret_EI = [0] * T
sum = 0
for t in range(T):

    # generate contexts
    for i in range(K):
        x[i] = sample_x()


    r_star = - float("inf")
    a_star  = 0
    for i in range(K):
        obj_func = np.transpose(x[i]) @ theta_hat
        if obj_func >  r_star :
            r_star  = obj_func
            a_star = i

    max = - float("inf")
    a_t = 0
    a_line = 0
    for i in range(K):
        x_invA_norm_sq = np.dot(x[i], invA) @ np.transpose(x[i])
        s_t = np.sqrt(x_invA_norm_sq)

        acq_value = 0
        if s_t != 0:
            z = (np.transpose(x[i]) @ theta_hat - r_star) / s_t
            acq_value = (np.transpose(x[i]) @ theta_hat - r_star) * norm.cdf(z) + s_t * norm.pdf(z)

        if acq_value > max:
            max = acq_value
            a_line = i

    print(a_line)

    if max >= 1/(t+1)**2:
        a_t = a_line
    else:
        a_t = a_star


    '''compute the reward'''
    reward = np.dot(x[a_t], theta_star) + R * ra.normal(0, 2)

    max_reward = - float("inf")
    for i in range(K):
        if np.max(np.dot(x[i], theta_star)) > max_reward:
            max_reward = np.max(np.dot(x[i], theta_star))
    sum = sum + max_reward - np.dot(x[a_t], theta_star)
    regret_EI[t] = sum
    '''update information for action a_t'''


    b += reward * x[a_t]
    A += np.outer(x[a_t], x[a_t])
    invA = la.inv(A)

    theta_hat = np.dot(invA, b)


import matplotlib.pyplot as plt
plt.figure()
plt.grid(True)


X_index = np.arange(1, T + 1)

plt.plot(X_index, regret_EI, color='red', marker='x', linewidth = 3.0)


plt.xlabel('Iterations', fontsize=40)
plt.ylabel('Cumulative Regret', fontsize=40)
plt.xticks(fontsize=35, rotation=0)
plt.yticks(fontsize=35, rotation=0)
#plt.title('XGBoost', fontsize=35, fontweight='bold')
plt.gca().legend(('LinEI'), prop={'size': 35})
plt.show();