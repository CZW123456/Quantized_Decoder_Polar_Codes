//
// Created by Zhiwei Cao on 2020/6/11.
//

#ifndef _LIBQUANTIZER_LLRMIQUANTIZER_H
#define _LIBQUANTIZER_LLRMIQUANTIZER_H

#endif //_LIBQUANTIZER_LLRMIQUANTIZER_H

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
using namespace cv;
namespace py = pybind11;

class LLRMIQuantizer{
public:
    LLRMIQuantizer() = default;
    py::tuple find_quantizer_decoder(py::array_t<double> &llr_density, py::array_t<int> &llr_quanta, int M, int K);
private:
    double compute_partial_mutual_information(Mat& density);
    Mat precompute_partial_mutual_information_table(Mat& density, int M, int K);
};