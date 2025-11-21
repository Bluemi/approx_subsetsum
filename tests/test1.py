#!/usr/bin/env python3

from approx_subsetsum import process_data
import numpy as np


def main():
    samples = np.array([1, 2, 3])
    capacity = 3
    res = process_data(samples, capacity)
    print(res)


if __name__ == '__main__':
    main()

