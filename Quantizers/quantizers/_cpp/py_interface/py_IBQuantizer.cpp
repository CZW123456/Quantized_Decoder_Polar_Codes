//
// Created by Zhiwei Cao on 2020/5/14.
//
#include <pybind11/pybind11.h>
#include "../IBQuantizers/IBQuantizers.h"

namespace py = pybind11;

void init_ModifiedsIB_quantizer(py::module& m){
    py::class_<ModifiedsIBQuantizer>(m, "ModifiedsIBQuantizer", "Modified Sequential Information Bottleneck Quantizer For Decoding Polar Codes")
            .def(py::init<double, int>(), py::arg("beta"), py::arg("num_run"))
            .def("find_quantizer", &ModifiedsIBQuantizer::find_quantizer, py::arg("p_y_given_x"), py::arg("border_vector_"), py::arg("K"));
}

