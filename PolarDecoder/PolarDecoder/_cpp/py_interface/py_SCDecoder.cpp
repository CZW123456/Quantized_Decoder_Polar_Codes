//
// Created by Zhiwei Cao on 2020/3/25.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SCDecoder.h"
namespace py = pybind11;
void init_SCDecoder(py::module& m){
    py::class_<SC>(m, "SCDecoder","Successive Cancellation Decoder")
            .def(py::init<int, int, vector<int>, vector<int>>(), py::arg("N"), py::arg("K"), py::arg("frozen_bits"), py::arg("message_bits"))
            .def("decode", &SC::decode, py::arg("llr"));
}

