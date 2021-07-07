//
// Created by Zhiwei Cao on 2020/3/28.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SCLDecoder.h"
namespace py = pybind11;
void init_SCLDecoder(py::module& m){
    py::class_<SCL>(m, "SCLDecoder", "Successive Cancellation List Decoder")
            .def(py::init<int, int, int, vector<int>, vector<int>>(), py::arg("N"), py::arg("K"), py::arg("L"), py::arg("frozen_bits"), py::arg("message_bits"))
            .def("decode", &SCL::decode, py::arg("llr"));
}
