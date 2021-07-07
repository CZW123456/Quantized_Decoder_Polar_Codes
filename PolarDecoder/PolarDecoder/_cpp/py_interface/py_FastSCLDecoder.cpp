//
// Created by Zhiwei Cao on 2020/3/28.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FastSCLDecoder.h"
namespace py = pybind11;
void init_FastSCLDecoder(py::module& m){
    py::class_<FastSCL>(m, "FastSCLDecoder", "Fast Successive Cancellation List Decoder Using Decoding Tree Prunining")
            .def(py::init<int, int, int, vector<int>, vector<int>, vector<int>>(), py::arg("N"), py::arg("K"), py::arg("L"), py::arg("frozen_bits"), py::arg("message_bits"), py::arg("node_type"))
            .def("decode", &FastSCL::decode, py::arg("llr"));
}