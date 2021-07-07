//
// Created by Zhiwei Cao on 2020/3/26.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FastSCDecoder.h"
namespace py = pybind11;
using namespace std;
void init_FastSCDecoder(py::module& m){
    py::class_<FastSC>(m, "FastSCDecoder","Fast Successive Cancellation Decoder By Decoding Tree Pruning")
            .def(py::init<int, int , vector<int> , vector<int>, vector<int>>(), py::arg("N"), py::arg("K"), py::arg("frozen_bits"), py::arg("message_bits"), py::arg("node_type"))
            .def("decode", &FastSC::decode, py::arg("llr"));
}
