//
// Created by Zhiwei Cao on 2020/6/11.
//

#include <pybind11/pybind11.h>
#include "../LLRMIQuantizer/LLRMIQuantizer.h"

namespace py = pybind11;

void init_LLRMI_quantizer(py::module& m){
    py::class_<LLRMIQuantizer>(m, "LLRMIQuantizer", "Quantizer that maximize mutual information in LLR domain")
            .def(py::init<>())
            .def("find_quantizer_decoder", &LLRMIQuantizer::find_quantizer_decoder, py::arg("llr_density"), py::arg("permutation"), py::arg("M"), py::arg("K"));
}
