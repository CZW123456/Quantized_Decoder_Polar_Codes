#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;

void init_DMetricCalculator(py::module& m);
void init_CASCLDecoder(py::module& m);

PYBIND11_MODULE(libPolarBD, m) {
    NDArrayConverter::init_numpy();
    m.doc() = "Polar Codes Blind Detection Related Modules";  // optional
    init_DMetricCalculator(m);
    init_CASCLDecoder(m);
}



