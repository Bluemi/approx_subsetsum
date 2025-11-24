#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <exception>
#include <chrono>

namespace py = pybind11;

class TimeoutException : public std::runtime_error {
public:
  // The constructor calls the base class constructor with a descriptive message
  explicit TimeoutException()
      : std::runtime_error("timeout") {}
};

// --- 1. The Core Templated Function ---
// This function is called by both the NumPy array and Python list handlers.
template<typename T>
py::array_t<std::uint32_t> subsetsum_impl(const T *data, py::ssize_t size, std::uint32_t capacity, const float timeout) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // dp[s] = index of element last used to reach sum s
  // -1 = unreachable, -2 = base for sum 0
  std::unordered_map<std::uint64_t, std::int64_t> dp;
  dp[0] = -2;

  for (py::ssize_t i = 0; i < size; i++) {
    T w = data[i];
    if (w > static_cast<T>(capacity))
      continue;
    std::vector<std::tuple<std::uint64_t, py::ssize_t>> changes;
    for (auto [s, _]: dp) {
      if (s > capacity - w) continue;
      if (timeout > 0.f && s % 1048576 == 0) {
        auto curr_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start_time).count() > timeout * 1000) {
          throw TimeoutException();
        }
      }
      auto search2 = dp.find(s + w);
      if (search2 == dp.end()) {
        changes.push_back({s + w, i});
      }
    }
    for (auto change : changes) {
      dp[std::get<0>(change)] = std::get<1>(change);
    }
  }

  // best achievable sum <= capacity
  T best = 0;
  bool found = false;
  for (py::ssize_t s = capacity; s >= 0; s--) {
    auto search = dp.find(s);
    if (search != dp.end()) {
      best = search->first;
      found = true;
      break;
    }
  }

  std::vector<std::uint32_t> indices;
  if (!found) {
    throw std::runtime_error("Failed to find solution.");
  }

  // reconstruct indices
  T s = best;
  while (s != 0) {
    int i = dp[s];
    indices.push_back(i);
    s -= data[i];

    auto curr_time = std::chrono::high_resolution_clock::now();
    if (timeout > 0.f && std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start_time).count() > timeout * 1000) {
      throw TimeoutException();
    }
  }

  return py::array_t<std::uint32_t>(indices.size(), indices.data(), py::capsule());
}

// --- 2. Helper to process NumPy array (dispatches) ---
template<typename T>
py::array_t<std::uint32_t> process_numpy_array(py::array_t<T> array, std::uint32_t capacity, float timeout) {
  // Ensure the array is 1D for simplicity
  if (array.ndim() != 1) {
    throw py::value_error("NumPy array must be 1-dimensional.");
  }
  // Get a pointer to the data and its size. No data copy occurs here.
  const T *data_ptr = array.data();
  py::ssize_t size = array.size();

  // Call the core templated function
  return subsetsum_impl<T>(data_ptr, size, capacity, timeout);
}

// --- 3. The main Python-facing function (Type Switch) ---
py::array_t<std::uint32_t> subsetsum(py::object data, std::uint32_t capacity, float timeout) {
  // Check if the input is a NumPy array (py::array)
  if (py::isinstance<py::array>(data)) {
    py::array arr = data.cast<py::array>();
    py::dtype dtype = arr.dtype();

    // Dispatch based on the detected NumPy dtype.
    // We check common numerical types and pass the data pointer directly.
    if (dtype.is(py::dtype::of<double>())) {
      return process_numpy_array<double>(arr.cast<py::array_t<double> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::int64_t>())) {
      return process_numpy_array<std::int64_t>(arr.cast<py::array_t<std::int64_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::int32_t>())) {
      return process_numpy_array<std::int32_t>(arr.cast<py::array_t<std::int32_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::int16_t>())) {
      return process_numpy_array<std::int16_t>(arr.cast<py::array_t<std::int16_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::int8_t>())) {
      return process_numpy_array<std::int8_t>(arr.cast<py::array_t<std::int16_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::uint64_t>())) {
      return process_numpy_array<std::uint64_t>(arr.cast<py::array_t<std::uint64_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::uint32_t>())) {
      return process_numpy_array<std::uint32_t>(arr.cast<py::array_t<std::uint32_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::uint16_t>())) {
      return process_numpy_array<std::uint16_t>(arr.cast<py::array_t<std::uint16_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<std::uint8_t>())) {
      return process_numpy_array<std::uint8_t>(arr.cast<py::array_t<std::uint16_t> >(), capacity, timeout);
    } else if (dtype.is(py::dtype::of<float>())) {
      return process_numpy_array<float>(arr.cast<py::array_t<float> >(), capacity, timeout);
    } else {
      throw py::value_error("Unsupported NumPy dtype detected.");
    }
  }
  // Check if the input is a standard Python list (py::list)
  else if (py::isinstance<py::list>(data)) {
    py::list lst = data.cast<py::list>();
    py::ssize_t size = py::len(lst);

    // **List Handling:** We must convert the Python list elements to a standard C++ type (e.g., double)
    // to get a contiguous data pointer needed by subsetsum_impl. This involves a copy/conversion.
    std::vector<double> cpp_vector;
    cpp_vector.reserve(size);

    for (py::handle item: lst) {
      try {
        // Cast Python object to double
        cpp_vector.push_back(item.cast<double>());
      } catch (const py::cast_error &e) {
        throw py::value_error("List must contain only numeric elements.");
      }
    }

    // Call the core templated function using the vector's data pointer
    return subsetsum_impl<double>(cpp_vector.data(), size, capacity, timeout);
  } else {
    throw py::value_error("Input must be a NumPy array or a Python list.");
  }
}

// --- 4. Pybind11 Module Definition ---
PYBIND11_MODULE(approx_subsetsum, m) {
  m.doc() = "Library to solve the subset sum problem.";

  py::register_exception<TimeoutException>(m, "TimeoutError", PyExc_TimeoutError);

  m.def(
    "subsetsum", &subsetsum, "Solves the subset sum problem.",
    py::arg("data"), py::arg("capacity"), py::arg("timeout") = -1.f
    );
}
