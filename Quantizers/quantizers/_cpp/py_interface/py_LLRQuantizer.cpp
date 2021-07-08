//
// Created by Zhiwei Cao on 2020/5/8.
//
#include <pybind11/pybind11.h>
#include "../LLRQuantizer/LLRQuantizer.h"

namespace py = pybind11;

void init_LLROptLS_quantizer(py::module& m){
    py::class_<LLROptLSQuantizer>(m, "LLRQuantizer", "Quantizer that minimize the quantization noise")
            .def(py::init<>())
            .def("find_OptLS_quantizer", &LLROptLSQuantizer::find_OptLS_quantizer, py::arg("llr_density"), py::arg("llr_quantas"), py::arg("M"), py::arg("K"));
}

