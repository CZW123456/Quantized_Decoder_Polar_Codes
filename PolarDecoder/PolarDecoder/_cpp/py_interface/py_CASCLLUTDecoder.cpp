//
// Created by Zhiwei Cao on 2020/6/7.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "CASCLLUTDecoder.h"
namespace py = pybind11;
void init_CASCLLUTDecoder(py::module& m){
    py::class_<CASCLLUT>(m, "CASCLLUTDecoder", "CRC Aided Successive Cancellation List LUT Decoder")
            .def(py::init<int, int, int, int, vector<int>, vector<int>, int, vector<int>&, vector<vector<vector<vector<int>>>>,
                    vector<vector<vector<vector<vector<int>>>>>,vector<vector<vector<double>>>>(),
                    py::arg("N"), py::arg("K"), py::arg("A"),
                    py::arg("L"), py::arg("frozen_bits"), py::arg("message_bits"), py::arg("crc_n"), py::arg("crc_p"),
                 py::arg("LUT_f"), py::arg("LUT_g"), py::arg("virtual_channel_llr"))
            .def("decode", &CASCLLUT::decode, py::arg("channel_quantized_symbols"));
}