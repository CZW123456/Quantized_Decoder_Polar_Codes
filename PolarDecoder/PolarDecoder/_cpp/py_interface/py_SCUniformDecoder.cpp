//
// Created by Zhiwei Cao on 2020/6/24.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SCUniformQuantizedDecoder.h"
namespace py = pybind11;

void init_SCUniformQuantizedDecoder(py::module& m){
    py::class_<SCUniformQuantizedDecoder>(m, "SCUniformQuantizedDecoder","Uniformly Quantized Successive Cancellation Decoder")
            .def(py::init<int, int, vector<int>, vector<int>, vector<double>&, vector<double>&, int>(),
                    py::arg("N"), py::arg("K"),py::arg("frozen_bits"),
                    py::arg("message_bits"), py::arg("decoder_r_f"),
                    py::arg("decoder_r_g"), py::arg("v"))
            .def("decode", &SCUniformQuantizedDecoder::decode, py::arg("llr"));
}

