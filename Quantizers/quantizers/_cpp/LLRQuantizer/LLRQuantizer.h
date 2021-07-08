//
// Created by Zhiwei Cao on 2020/5/8.
//

#ifndef _LIBQUANTIZER_LLRQUANTIZER_H
#define _LIBQUANTIZER_LLRQUANTIZER_H

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
using namespace cv;
namespace py = pybind11;

class LLROptLSQuantizer{
public:
    LLROptLSQuantizer() = default;
    py::tuple find_OptLS_quantizer(py::array_t<double> &llr_density, py::array_t<double> &llr_quanta, int M, int K);
private:
    double compute_partial_quantization_noise(Mat& density, Mat& quanta);
    Mat precompute_quantization_noise_table(Mat& density, Mat& quantas, int M, int K);
};

#endif //_LIBQUANTIZER_LLRQUANTIZER_H



