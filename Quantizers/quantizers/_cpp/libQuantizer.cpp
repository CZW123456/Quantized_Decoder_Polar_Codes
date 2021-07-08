#include <pybind11/pybind11.h>
#include "MMIQuantizer/MMIQuantizer.h"
#include "ndarray_converter.h"

namespace py = pybind11;

void init_MMI_quantizer(py::module& m);
void init_LLROptLS_quantizer(py::module& m);
void init_ModifiedsIB_quantizer(py::module& m);
void init_LLRMI_quantizer(py::module& m);

PYBIND11_MODULE(_libQuantizer, m) {
    NDArrayConverter::init_numpy();
    m.doc() = "MMI Quantizer For Polar Codes";  // optional
    init_MMI_quantizer(m);
    init_LLROptLS_quantizer(m);
    init_ModifiedsIB_quantizer(m);
    init_LLRMI_quantizer(m);
}


