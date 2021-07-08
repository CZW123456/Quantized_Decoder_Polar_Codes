#include <pybind11/pybind11.h>
#include "../MMIQuantizer/MMIQuantizer.h"

namespace py = pybind11;

void init_MMI_quantizer(py::module& m){
    py::class_<MMIQuantizer>(m, "MMIQuantizer", "MMI Quantizer For Decoding Polar Codes")
            .def(py::init<float, float>(), py::arg("px1"), py::arg("px_minus1"))
            .def("find_opt_quantizer_consider_unique", &MMIQuantizer::find_opt_quantizer_consider_unique, py::arg("joint_probs"), py::arg("K"), py::arg("mode"))
            .def("find_opt_quantizer", &MMIQuantizer::find_opt_quantizer, py::arg("joint_probs"), py::arg("K"))
            .def("find_opt_quantizer_AWGN", &MMIQuantizer::find_opr_quantizer_AWGN, py::arg("joint_probs"), py::arg("K"));
}

