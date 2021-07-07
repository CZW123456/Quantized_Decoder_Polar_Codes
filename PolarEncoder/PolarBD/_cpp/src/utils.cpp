//
// Created by Zhiwei Cao on 2021/3/17.
//

#include "utils.h"

void f(double *out, const double *a, const double *b, const int l){
    for (int i = 0; i < l; ++i) {
        out[i] = (sign(a[i])) * (sign(b[i])) * min(abs(a[i]), abs(b[i]));
    }
}

void g(double *out, const double *a, const double *b, const u_int8_t *u, const int l){
    for (int i = 0; i < l; ++i) {
        out[i] = (1 - 2 * u[i]) * a[i] + b[i];
    }
}

void u(u_int8_t *out, const u_int8_t *a, const u_int8_t *b, const int l){
    for (int i = 0; i < l; ++i) {
        out[i] = a[i] ^ b[i];
    }
    memcpy(out + l, b, sizeof(uint8_t) * l);
}