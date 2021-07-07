//
// Created by Zhiwei Cao on 2020/6/24.
//

#ifndef _LIBPOLARDECODER_SCUNIFORMQUANTIZEDDECODER_H
#define _LIBPOLARDECODER_SCUNIFORMQUANTIZEDDECODER_H
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "utils.h"
namespace py = pybind11;
using namespace std;

class SCUniformQuantizedDecoder {
public:
    SCUniformQuantizedDecoder(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_, vector<double> &_decoder_r_f, vector<double> &_decoder_r_g, int _v);
    py::array_t<uint8_t> decode(py::array_t<double>& llr);

private:
    int N;
    int K;
    int v;
    vector<int> frozen_bits;
    vector<int> message_bits;
    vector<double> decoder_r_f;
    vector<double> decoder_r_g;

};

#endif //_LIBPOLARDECODER_SCUNIFORMQUANTIZEDDECODER_H
