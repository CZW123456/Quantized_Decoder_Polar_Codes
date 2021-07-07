//
// Created by Zhiwei Cao on 2021/3/17.
//

#ifndef LIBPOLARBD_UTILS_H
#define LIBPOLARBD_UTILS_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
using namespace std;
#define sign(x) (((x) < 0 )? -1 : ((x)> 0))

void f(double *out, const double *a, const double *b, int l);

void g(double *out, const double *a, const double *b, const u_int8_t *u, int l);

void u(u_int8_t *out, const u_int8_t *a, const u_int8_t *b, int l);

#endif //LIBPOLARBD_UTILS_H
