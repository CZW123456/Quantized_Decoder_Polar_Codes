//
// Created by Zhiwei Cao on 2020/3/26.
//

#ifndef _LIBPOLARDECODER_UTILS_H
#define _LIBPOLARDECODER_UTILS_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
using namespace std;
#define sign(x) (((x) < 0 )? -1 : ((x)> 0))

void f(double *out, const double *a, const double *b, int l);

void g(double *out, const double *a, const double *b, const u_int8_t *u, int l);

void q_f(double *out, const double *a, const double *b, int l, double r, double M);

void q_g(double *out, const double *a, const double *b, const u_int8_t *u, int l, double r, double M);

void non_uniform_q_f(double *out, const double *a, const double *b, int l, vector<double> &boundary, vector<double> &reconstruct);

void non_uniform_q_g(double *out, const double *a, const double *b, const u_int8_t *u, int l, vector<double> &boundary, vector<double> &reconstruct);

void u(u_int8_t *out, const u_int8_t *a, const u_int8_t *b, int l);

class CRC{
public:
    CRC();
    CRC(int _crc_n, std::vector<int> loc);
    vector<u_char> encoding(vector<u_char> &x);
private:
    int crc_n;
    vector<u_char> crc_p;
};

#endif //_LIBPOLARDECODER_UTILS_H
