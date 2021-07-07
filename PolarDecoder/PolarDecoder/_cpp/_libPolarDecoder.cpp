//
// Created by Zhiwei Cao on 2020/3/25.
//
#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;

void init_SCDecoder(py::module& m);
void init_SCLDecoder(py::module& m);
void init_CASCLDecoder(py::module& m);
void init_FastSCDecoder(py::module& m);
void init_FastSCLDecoder(py::module& m);

void init_SCLUTDecoder(py::module& m);
void init_SCLLUTDecoder(py::module& m);
void init_CASCLLUTDecoder(py::module& m);
void init_FastSCLUTDecoder(py::module& m);
void init_FastSCLLUTDecoder(py::module& m);
void init_CAFastSCLLUTDecoder(py::module& m);

void init_SCUniformQuantizedDecoder(py::module& m);
void init_SCLUniformQuantizedDecoder(py::module& m);

void init_SCLloydQuantizedDecoder(py::module& m);
void init_SCLLloydQuantizedDecoder(py::module& m);


PYBIND11_MODULE(_libPolarDecoder, m) {
    NDArrayConverter::init_numpy();
    m.doc() = "Decoders For Polar Codes";  // optional
    init_SCDecoder(m);
    init_FastSCDecoder(m);
    init_SCLUTDecoder(m);
    init_FastSCLUTDecoder(m);

    init_SCLDecoder(m);
    init_CASCLDecoder(m);
    init_FastSCLDecoder(m);
    init_SCLLUTDecoder(m);
    init_CASCLLUTDecoder(m);
    init_FastSCLLUTDecoder(m);
    init_CAFastSCLLUTDecoder(m);

    init_SCUniformQuantizedDecoder(m);
    init_SCLUniformQuantizedDecoder(m);

    init_SCLloydQuantizedDecoder(m);
    init_SCLLloydQuantizedDecoder(m);
}



