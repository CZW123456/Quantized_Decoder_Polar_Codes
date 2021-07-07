//
// Created by Zhiwei Cao on 2020/3/27.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FastSCLUT.h"
namespace py = pybind11;
using namespace std;
void init_FastSCLUTDecoder(py::module& m){
    py::class_<FastSCLUT>(m, "FastSCLUTDecoder","Fast Successive Cancellation Decoder Using LUTs")
            .def(py::init<int, int , vector<int> , vector<int>, vector<int>, vector<vector<vector<vector<int>>>>, vector<vector<vector<vector<vector<int>>>>>,
                    vector<vector<vector<double>>>>(), py::arg("N"), py::arg("K"), py::arg("frozen_bits"),
                    py::arg("message_bits"), py::arg("node_type"), py::arg("LUT_Fs"), py::arg("LUT_Gs"), py::arg("virtual_channel_llr"))
            .def("decode", &FastSCLUT::decode, py::arg("llr"));
}