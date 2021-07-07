//
// Created by Zhiwei Cao on 2021/1/14.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DMetric.h"

namespace py = pybind11;
void init_DMetricCalculator(py::module& m){
    py::class_<DMetricCalculator>(m, "DMetricCalculator","DMetric Calculator with Fast-SSC")
            .def(py::init<int, int , vector<int> , vector<int>, vector<int>>(), py::arg("N"), py::arg("K"), py::arg("frozen_bits"), py::arg("message_bits"), py::arg("node_type"))
            .def("calculate", &DMetricCalculator::calculate, py::arg("llr"));
}