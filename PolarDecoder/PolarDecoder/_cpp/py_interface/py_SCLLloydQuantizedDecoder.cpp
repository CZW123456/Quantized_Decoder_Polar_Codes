//
// Created by Zhiwei Cao on 2020/9/4.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SCLLloydQuantizedDecoder.h"
namespace py = pybind11;

void init_SCLLloydQuantizedDecoder(py::module& m){
    py::class_<SCLLloydQuantizedDecoder>(m, "SCLLloydQuantizedDecoder","Quantized Successive Cancellation List Decoder With Lloyd Algorithm")
            .def(py::init<int, int, int, vector<int>, vector<int>, vector <vector<double>> &, vector <vector<double>> &,
                         vector <vector<double>>&, vector <vector<double>>&, int>(),
                 py::arg("N"), py::arg("K"), py::arg("L"), py::arg("frozen_bits"),
                 py::arg("message_bits"), py::arg("boundaries_f"),
                 py::arg("boundaries_g"), py::arg("reconstruction_f"), py::arg("reconstruction_g"),
                 py::arg("v"))
            .def("decode", &SCLLloydQuantizedDecoder::decode, py::arg("llr"));
}