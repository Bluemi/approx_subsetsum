#!/usr/bin/env python3

import time
import numpy as np
from approx_subsetsum import subsetsum


def test_bench():
    samples = np.array([1, 2, 3, 4, 5])
    capacity = 7
    start_time = time.time()
    res = subsetsum(samples, capacity)
    end_time = time.time()
    solution = samples[res]
    summe = np.sum(solution)
    assert capacity == summe
    print(res)
    print(f'sum={summe}  capacity={capacity}  time={end_time - start_time:.3f}')


def test_benchmark():
    samples = np.random.randint(3, 10, 2000)
    capacity = int(np.sum(samples) * 0.8)
    start_time = time.time()
    res = subsetsum(samples, capacity)
    end_time = time.time()
    solution = samples[res]
    summe = np.sum(solution)
    print(res)
    print(f'sum={summe}  capacity={capacity}  time={end_time - start_time:.3f}')


def test_benchmark_no_solution():
    samples = np.random.randint(3, 10, 1000) * 2
    capacity = 801
    start_time = time.time()
    res = subsetsum(samples, capacity)
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

