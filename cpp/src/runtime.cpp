#include <pybind11/pybind11.h>

#include <atomic>
#include <stdexcept>

#include "native_fem/parallel.hpp"
#include "native_fem/python_bindings.hpp"

namespace {
std::atomic<int> thread_count{1};
}

int native_fem::num_threads() {
    return thread_count.load(std::memory_order_relaxed);
}

void native_fem::set_num_threads(int count) {
    if (count < 1) throw std::invalid_argument("thread count must be positive");
    thread_count.store(count, std::memory_order_relaxed);
}

void native_fem::bind_runtime(pybind11::module_& module) {
    module.def("get_num_threads", &num_threads);
    module.def("set_num_threads", &set_num_threads);
}
