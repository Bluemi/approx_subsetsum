# approx\_subsetsum

Find fast approximate solutions for the [subset sum problem](https://en.wikipedia.org/wiki/Subset_sum_problem).

## Usage
Given an integer numpy array `samples`, this function tries to find a subset of `samples`, that has a sum close to a given `capacity`.

```py
import numpy as np
from approx_subsetsum import subsetsum

samples = np.array([1, 2, 3, 4, 5])
capacity = 7
indices = subsetsum(samples, capacity)
solution = samples[res]
```

## Installation

```sh
pip install approx_subsetsum
```
