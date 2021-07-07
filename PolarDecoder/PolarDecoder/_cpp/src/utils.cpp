//
// Created by Zhiwei Cao on 2020/3/26.
//
#include "utils.h"

using namespace std;

inline double Q(double x, double r, double M){
    return abs(x) > M ? sign(x)*(M - 0.5*r) : (floor(x/r) + 0.5)*r;
}

inline double bisect(double a, vector<double> &boundary, vector<double> &reconstruct){
    int lo = 0;
    int hi = boundary.size();
    while (lo < hi){
        int mid = (lo + hi) / 2;
        if (boundary[mid] < a){
            lo = mid + 1;
        } else{
            hi = mid;
        }
    }
    return reconstruct[lo - 1];
}

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

void q_f(double *out, const double *a, const double *b, int l, double r, double M){
    for (int i = 0; i < l; ++i) {
        out[i] = Q((sign(a[i])) * (sign(b[i])) * min(abs(a[i]), abs(b[i])), r, M);
    }
}

void q_g(double *out, const double *a, const double *b, const u_int8_t *u, int l, double r, double M){
    for (int i = 0; i < l; ++i) {
        out[i] = Q((1 - 2 * u[i]) * a[i] + b[i], r, M);
    }
}

void non_uniform_q_f(double *out, const double *a, const double *b, int l, vector<double> &boundary, vector<double> &reconstruct){
    for (int i = 0; i < l; ++i) {
        out[i] = bisect((sign(a[i])) * (sign(b[i])) * min(abs(a[i]), abs(b[i])), boundary, reconstruct);
    }
}

void non_uniform_q_g(double *out, const double *a, const double *b, const u_int8_t *u, int l, vector<double> &boundary, vector<double> &reconstruct){
    for (int i = 0; i < l; ++i) {
        out[i] = bisect((1 - 2 * u[i]) * a[i] + b[i], boundary, reconstruct);
    }
}

void u(u_int8_t *out, const u_int8_t *a, const u_int8_t *b, const int l){
    for (int i = 0; i < l; ++i) {
        out[i] = a[i] ^ b[i];
    }
    memcpy(out + l, b, sizeof(uint8_t) * l);
}

CRC::CRC(int _crc_n, std::vector<int> loc) {
    crc_n = _crc_n;
    crc_p.resize(crc_n + 1, 0);
    for (int i = 0; i < loc.size(); ++i) {
        crc_p[loc[i]] = 1;
    }
}

vector<uint8_t> CRC::encoding(vector<uint8_t> &info) {
    int info_length = info.size();
    int times = info_length;
    int n = crc_n + 1;
    vector<uint8_t> u(info_length + crc_n, 0);
    memcpy((uint8_t*)u.data(), (uint8_t*)info.data(), info_length);
    for (int i = 0; i < times; ++i) {
        if (u[i] == 1) {
            for (int j = 0; j < n; ++j) {
                u[j + i] = (u[j + i] + crc_p[j]) % 2;
            }
        }
    }
    vector<uint8_t> check_code(crc_n);
    memcpy(check_code.data(), u.data() + info_length, sizeof(uint8_t)*crc_n);
    return check_code;
}