import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
from typing import List

sns.set()


def make_data(n: int, f: float = 0.3, seed: int = 1) -> np.ndarray:
    rand = np.random.RandomState(seed)
    x = rand.randn(n)
    x[int(f * n):] += 5
    return x


if __name__ == '__main__':
    data = make_data(1000)
    hist = plt.hist(data, bins=30, density=True)
    plt.show()
