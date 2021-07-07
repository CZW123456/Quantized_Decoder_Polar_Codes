//
// Created by Zhiwei Cao on 2020/3/28.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SCLLUTDecoder.h"

namespace py = pybind11;
void init_SCLLUTDecoder(py::module& m){
    py::class_<SCLLUT>(m, "SCLLUTDecoder", "Successive Cancellation Decoder Using LUT")
            .def(py::init<int, int, int, vector<int>, vector<int>, vector<vector<vector<vector<int>>>>, vector<vector<vector<vector<vector<int>>>>>,
                         vector<vector<vector<double>>>>(), py::arg("N"), py::arg("K"), py::arg("L"), py::arg("frozen_bits"), py::arg("message_bits"),
                 py::arg("LUT_f"), py::arg("LUT_g"), py::arg("virtual_channel_llr"))
            .def("decode", &SCLLUT::decode, py::arg("channel_quantized_symbols"));
}