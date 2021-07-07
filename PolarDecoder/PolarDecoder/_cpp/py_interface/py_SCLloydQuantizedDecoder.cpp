//
// Created by Zhiwei Cao on 2020/9/4.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SCLloydQuantizedDecoder.h"
namespace py = pybind11;

void init_SCLloydQuantizedDecoder(py::module& m){
    py::class_<SCLloydQuantizedDecoder>(m, "SCLloydQuantizedDecoder","Quantized Successive Cancellation Decoder With Lloyd Algorithm")
            .def(py::init<int, int, vector<int>, vector<int>, vector <vector<double>> &, vector <vector<double>> &,
                    vector <vector<double>>&, vector <vector<double>>&, int>(),
                 py::arg("N"), py::arg("K"),py::arg("frozen_bits"),
                 py::arg("message_bits"), py::arg("boundaries_f"),
                 py::arg("boundaries_g"), py::arg("reconstruction_f"), py::arg("reconstruction_g"),
                 py::arg("v"))
            .def("decode", &SCLloydQuantizedDecoder::decode, py::arg("llr"));
}