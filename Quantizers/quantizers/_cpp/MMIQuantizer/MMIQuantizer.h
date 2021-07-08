//
// Created by Zhiwei Cao on 2020/3/11.
//

#ifndef POLARQUANTIZAER_MMIQUANTIZAER_H
#define POLARQUANTIZAER_MMIQUANTIZAER_H

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
using namespace std;
using namespace cv;
namespace py = pybind11;
class MMIQuantizer{
public:
    MMIQuantizer(double p1, double p_minus_1);
    inline double compute_partial_entropy(Mat &pyx, Mat &pyx_minus_1);
    Mat precompute_partial_entropy_table(int M, int K);
    py::tuple find_opt_quantizer_consider_unique(py::array_t<double> &joint_prob_ndarrary, int K, const string& mode="f");
    py::tuple find_opt_quantizer(py::array_t<double> &joint_prob_ndarrary, int K);
    py::array_t<int> find_opr_quantizer_AWGN(py::array_t<double> &joint_prob_ndarrary, int K);
private:
    float m_p1;
    float m_p_minus_1;
    Mat pyx_1;
    Mat pyx_minus1;
};
#endif //POLARQUANTIZAER_MMIQUANTIZAER_H
