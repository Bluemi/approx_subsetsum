#!/usr/bin/env python3

import time
import numpy as np
from approx_subsetsum import subsetsum, TimeoutError


def main():
    # for n in [10000, 50000, 100000, 500000, 1000000, 5000000]:
    #     for c in [10000, 50000, 100000, 500000, 1000000]:
    #         bench(n, c)

    for n in np.logspace(4, 6, 8):
        for c in np.logspace(4, 5.5, 8):
            bench(int(n), int(c))


def bench(n_samples, c):
    samples = np.random.randint(1, 1000000, n_samples)
    start_time = time.time()
    try:
        res = subsetsum(samples, c, 10.0)
        end_time = time.time()
        solution = samples[res]
        summe = np.sum(solution)
        print(f'[{n_samples:>10d},  {c:>10d}, {end_time - start_time:.3f}],')
    except TimeoutError:
        print(f'[{n_samples:>10d},  {c:>10d}, 10],')


if __name__ == '__main__':
    main()
