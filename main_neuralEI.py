import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bandit import ContextualBandit
from neural_ei import NeuralEI
sns.set()

T = int(5e2)
n_arms = 4
n_features = 8
noise_std = 0.1

confidence_scaling_factor = noise_std

n_sim = 5

SEED = 42
np.random.seed(SEED)

p = 0.2
hidden_size = 32
epochs = 100
train_every = 10
use_cuda = False

a = np.random.randn(n_features)
a /= np.linalg.norm(a, ord=2)
reward_func = lambda x: 10 * np.dot(a, x)

bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=noise_std, seed=SEED)

regrets = np.empty((n_sim, T))

for i in range(n_sim):
    bandit.reset_rewards()
    model = NeuralEI(bandit,
                      hidden_size=hidden_size,
                      reg_factor=1.0,
                      delta=0.1,
                      confidence_scaling_factor=confidence_scaling_factor,
                      training_window=100,
                      p=p,
                      learning_rate=0.01,
                      epochs=epochs,
                      train_every=train_every,
                      use_cuda=use_cuda
                      )

    model.run()
    regrets[i] = np.cumsum(model.regrets)

fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)

t = np.arange(T)

mean_regrets = np.mean(regrets, axis=0)
std_regrets = np.std(regrets, axis=0) / np.sqrt(regrets.shape[0])
ax.plot(mean_regrets)
ax.fill_between(t, mean_regrets - 2 * std_regrets, mean_regrets + 2 * std_regrets, alpha=0.15)
ax.set_title('Cumulative regret')

plt.tight_layout()
plt.show()