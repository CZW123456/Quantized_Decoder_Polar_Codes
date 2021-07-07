//
// Created by Zhiwei Cao on 2021/3/25.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "CASCLWithRNTI.h"
namespace py = pybind11;
void init_CASCLDecoder(py::module& m){
    py::class_<CASCL>(m, "CASCLDecoder", "CRC Aided Successive Cancellation List Decoder")
            .def(py::init<int, int, int, int, vector<int>&, vector<int>&, int, vector<int>&>(), py::arg("N"), py::arg("K"), py::arg("A"), py::arg("L"), py::arg("frozen_bits"), py::arg("message_bits"), py::arg("crc_n"), py::arg("crc_p"))
            .def("decode", &CASCL::decode, py::arg("llr"), py::arg("RNTI"));
}