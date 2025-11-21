#!/usr/bin/env python3

import time
import numpy as np
from approx_subsetsum import subsetsum

def subset_sum_closest(samples, n):
    samples = np.asarray(samples, dtype=int)

    # dp[s] = index of element last used to reach sum s
    # -1 = unreachable, -2 = base for sum 0
    dp = np.full(n + 1, -1, dtype=int)
    dp[0] = -2

    for i, w in enumerate(samples):
        if w > n:
            continue
        # go backwards to avoid reusing the same element
        for s in range(n - w, -1, -1):
            if dp[s] != -1 and dp[s + w] == -1:
                dp[s + w] = i

    # best achievable sum <= N
    best = -1
    for s in range(n, -1, -1):
        if dp[s] != -1:
            best = s
            break
    if best == -1:  # only possible if all samples > N
        return []

    # reconstruct indices
    indices = []
    s = best
    while s != 0:
        i = dp[s]
        indices.append(i)
        s -= samples[i]
    indices.reverse()
    return indices


def benchmark(func):
    samples = np.random.randint(3, 10, 2000)
    capacity = int(np.sum(samples) * 0.8)
    start_time = time.time()
    res = func(samples, capacity)
    end_time = time.time()
    solution = samples[res]
    summe = np.sum(solution)
    print(res)
    print(f'sum={summe}  capacity={capacity}  time={end_time - start_time:.3f}')

def benchmark_no_solution(func):
    samples = np.random.randint(3, 10, 1000) * 2
    capacity = 801
    start_time = time.time()
    res = func(samples, capacity)
    end_time = time.time()
    solution = samples[res]
    summe = np.sum(solution)
    print(res)
    print(f'sum={summe}  capacity={capacity}  time={end_time - start_time:.3f}')



def main():
    benchmark_no_solution(subsetsum)
    benchmark_no_solution(subset_sum_closest)


if __name__ == '__main__':
    main()

