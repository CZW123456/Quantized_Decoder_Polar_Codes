//
// Created by Zhiwei Cao on 2020/5/14.
//

#ifndef _LIBQUANTIZER_IBQUANTIZERS_H
#define _LIBQUANTIZER_IBQUANTIZERS_H

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/core.hpp>

namespace py = pybind11;
using namespace std;
using namespace cv;

class ModifiedsIBQuantizer{
public:
    ModifiedsIBQuantizer(double beta_, int nrun_);
    py::tuple find_quantizer(py::array_t<double> &pxy, py::array_t<int> &border_vector, int K);
private:
    static vector<double> compute_merge_cost(Mat& p_t, Mat& p_x_given_t, int border_between_cluster);
    double beta;
    int nrun;
};
#endif //_LIBQUANTIZER_IBQUANTIZERS_H
