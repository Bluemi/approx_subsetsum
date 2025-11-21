#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

// --- 1. The Core Templated Function ---
// This function is called by both the NumPy array and Python list handlers.
template<typename T>
py::array_t<std::uint32_t> foo_impl(const T* data, py::ssize_t size, std::uint32_t capacity) {
	std::cout << 8 << std::endl;

    // dp[s] = index of element last used to reach sum s
    // -1 = unreachable, -2 = base for sum 0
    std::vector<int> dp(capacity+1, -1);
    dp[0] = -2;

	for (py::ssize_t i = 0; i < size; i++) {
		T w = data[i];
		if (w > capacity)
			continue;
		for (py::ssize_t s = capacity - w; s >= 0; s--) {
			if (dp[s] != -1 && dp[s + w] == -1) {
				dp[s+w] = i;
			}
		}
	}
	std::cout << 9 << std::endl;
	/*
    for i, w in enumerate(samples):
        if w > capacity:
            continue
        # go backwards to avoid reusing the same element
        for s in range(capacity - w, -1, -1):
            if dp[s] != -1 and dp[s + w] == -1:
                dp[s + w] = i
	*/
	std::cout << 10 << std::endl;

    // best achievable sum <= capacity
    T best = -1;
	for (py::ssize_t s = capacity; s >= 0; s--) {
		if (dp[s] != -1) {
			best = s;
			break;
		}
	}
	std::cout << 11 << std::endl;

	std::vector<std::uint32_t> indices;
	if (best == -1) {
		return py::cast(indices);
	}
	/*
    for s in range(capacity, -1, -1):
        if dp[s] != -1:
            best = s
            break
    if best == -1:  # only possible if all samples > capacity
        return [], 0
	*/

    // reconstruct indices
	/*
    indices = []
    s = best
    while s != 0:
        i = dp[s]
        indices.append(i)
        s -= samples[i]
    indices.reverse()
    return indices, best
	*/
	std::cout << 12 << std::endl;
	T s = best;
	while (s != 0) {
		int i = dp[s];
		indices.push_back(i);
		s -= data[i];
	}
	std::cout << 13 << std::endl;
	std::cout << "len indices: " << indices.size() << std::endl;

	auto res = py::cast(indices); 
	std::cout << 14 << std::endl;
	return res;
}

// --- 2. Helper to process NumPy array (dispatches) ---
template<typename T>
py::array_t<std::uint32_t> process_numpy_array(py::array_t<T> array, std::uint32_t capacity) {
	std::cout << 5 << std::endl;
    // Ensure the array is 1D for simplicity
    if (array.ndim() != 1) {
        throw std::runtime_error("NumPy array must be 1-dimensional.");
    }
	std::cout << 6 << std::endl;
    // Get a pointer to the data and its size. No data copy occurs here.
    const T* data_ptr = array.data();
    py::ssize_t size = array.size();
	std::cout << 7 << std::endl;

    // Call the core templated function
    py::array_t<std::uint32_t> res = foo_impl<T>(data_ptr, size, capacity);
	std::cout << 7.1 << std::endl;
	return res;
}

// --- 3. The main Python-facing function (Type Switch) ---
py::array_t<std::uint32_t> process_data(py::object data, std::uint32_t capacity) {
	std::cout << 1 << std::endl;
    // Check if the input is a NumPy array (py::array)
    if (py::isinstance<py::array>(data)) {
		std::cout << 2 << std::endl;
        py::array arr = data.cast<py::array>();
        py::dtype dtype = arr.dtype();
		std::cout << 3 << std::endl;

        // Dispatch based on the detected NumPy dtype.
        // We check common numerical types and pass the data pointer directly.
        if (dtype.is(py::dtype::of<double>())) {
            return process_numpy_array<double>(arr.cast<py::array_t<double>>(), capacity);
        } else if (dtype.is(py::dtype::of<std::int64_t>())) {
			std::cout << 4 << std::endl;
            auto res = process_numpy_array<std::int64_t>(arr.cast<py::array_t<std::int64_t>>(), capacity);
			std::cout << 4.1 << std::endl;
			return res;
        } else if (dtype.is(py::dtype::of<float>())) {
            return process_numpy_array<float>(arr.cast<py::array_t<float>>(), capacity);
        } else if (dtype.is(py::dtype::of<std::int32_t>())) {
            return process_numpy_array<std::int32_t>(arr.cast<py::array_t<std::int32_t>>(), capacity);
        } else {
             throw std::runtime_error("Unsupported NumPy dtype detected.");
        }
		// TODO: add more data types
    }
    // Check if the input is a standard Python list (py::list)
    else if (py::isinstance<py::list>(data)) {
        py::list lst = data.cast<py::list>();
        py::ssize_t size = py::len(lst);
        
        // **List Handling:** We must convert the Python list elements to a standard C++ type (e.g., double)
        // to get a contiguous data pointer needed by foo_impl. This involves a copy/conversion.
        std::vector<double> cpp_vector;
        cpp_vector.reserve(size);

        for (py::handle item : lst) {
            try {
                // Cast Python object to double
                cpp_vector.push_back(item.cast<double>());
            } catch (const py::cast_error& e) {
                throw std::runtime_error("List must contain only numeric elements.");
            }
        }
        
        // Call the core templated function using the vector's data pointer
        return foo_impl<double>(cpp_vector.data(), size, capacity);
    }
    else {
        throw std::runtime_error("Input must be a NumPy array or a Python list.");
    }
}

// --- 4. Pybind11 Module Definition ---
PYBIND11_MODULE(approx_subsetsum, m) {
    m.doc() = "pybind11 data processor that handles numpy arrays and python lists."; // Optional module docstring

    m.def("process_data", &process_data,
          "Processes data from either a NumPy array or a Python list, dispatching to a templated C++ function.");
}
