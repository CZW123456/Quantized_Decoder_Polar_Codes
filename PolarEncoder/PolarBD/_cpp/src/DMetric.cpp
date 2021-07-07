//
// Created by Zhiwei Cao on 2021/3/17.
//

#include "DMetric.h"
#include <utility>

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return distance(first, min_element(first, last));
}

DMetricCalculator::DMetricCalculator(int N_, int K_, vector<int> frozen_bits_, vector<int> message_bits_,
                                     vector<int> node_type_) {
    N = N_;
    K = K_;
    frozen_bits = std::move(frozen_bits_);
    message_bits = std::move(message_bits_);
    node_type = std::move(node_type_);
}


double DMetricCalculator::calculate(py::array_t<double> &llr)  {
    int n = (int)log2f((float)N);
    int depth = 0;
    int node = 0;
    bool done = false;
    py::buffer_info buf = llr.request();
    vector<ssize_t> size_buf = buf.shape;
    assert(buf.shape[0] == 1);
    std::vector<int> node_state(2 * N - 1);
    std::vector<double> alpha((n + 1) * N);
    std::vector<u_int8_t> ucap((n + 1) * N);
    memcpy(alpha.data(), (double*)llr.data(), sizeof(double) * N);
    double DMetric = 0.f;
    while (!done){
        if (depth == n){
            if (frozen_bits[node] == 1){
                ucap[n*N+node] = 0;
            } else{
                ucap[n*N+node] = (uint8_t)(alpha[n*N+node] <= 0);
            }
            node /= 2;
            depth--;
        } else{
            int node_posi = (1 << depth) + node - 1;
            if (node_state[node_posi] == 0){
                if (node_type[node_posi] == 0) {
                    // R0 node
                    int temp = 1 << (n - depth);
                    auto *pucap = (u_int8_t*)ucap.data() + (depth*N+temp*node);
                    double* palpha = alpha.data() + (depth*N+temp*node);
                    memset(pucap, 0, sizeof(uint8_t) * temp);
                    node /= 2;
                    depth--;
                    double tmp = 0;
                    // update DMetric for Rate-0 node
                    for (int i = 0; i < temp; ++i) {
                        tmp += palpha[i];
                    }
                    DMetric += tmp/temp;
//                    tmp = tmp/temp;
//                    DMetric += 0.1 * pow(tmp, 2);
                    continue;
                }
                if (node_type[node_posi] == 1){
                    // R1 node
                    int temp = 1 << (n - depth);
                    double* palpha = alpha.data() + (depth*N+temp*node);
                    uint8_t *pucap = ucap.data() + (depth*N+temp*node);
                    for (int i = 0; i < temp; ++i) {
                        pucap[i] = palpha[i] <= 0;
                    }
                    node /= 2;
                    depth--;
                    continue;
                }
                if (node_type[node_posi] == 2){
                    // Rep node
                    int temp = 1 << (n - depth);
                    double* palpha = alpha.data() + (depth*N+temp*node);
                    uint8_t *pucap = ucap.data() + (depth*N+temp*node);
                    double S = 0;
                    for (int i = 0; i < temp; ++i) {
                        S += palpha[i];
                    }
                    memset(pucap, (uint8_t)(S <= 0), sizeof(uint8_t)*temp);
                    node /= 2;
                    depth--;
                    double tmp = 0;
                    // update DMetric for REP node
                    for (int i = 0; i < temp; ++i) {
                        tmp += palpha[i];
                    }
                    tmp = abs(tmp)/temp;
                    DMetric += tmp;
//                    DMetric += 1 / (1 + exp(-abs(tmp)/temp));
//                    DMetric += 0.1 * pow(tmp, 2);
                    continue;
                }
                if (node_type[node_posi] == 3){
                    // SPC node
                    int temp = 1 << (n - depth);
                    double* palpha = alpha.data() + (depth*N+temp*node);
                    uint8_t *pucap = ucap.data() + (depth*N+temp*node);
                    auto *HD = new uint8_t[temp];
                    std::vector<double> abs_alpha(temp);
                    auto *pabs = abs_alpha.data();
                    int parity_check = 0;
                    for (int i = 0; i < temp; ++i) {
                        HD[i] = (uint8_t)(palpha[i] <= 0);
                        parity_check += (int)HD[i];
                        pabs[i] = abs(palpha[i]);
                    }
                    parity_check %= 2;
                    if (!parity_check){
                        memcpy(pucap, HD, sizeof(uint8_t)*temp);
                    } else{
                        size_t min_idx = argmin(abs_alpha.begin(), abs_alpha.end());
                        HD[min_idx] = 1 - HD[min_idx];
                        memcpy(pucap, HD, sizeof(uint8_t)*temp);
                    }
                    node /= 2;
                    depth -= 1;
//                    size_t min_idx = argmin(abs_alpha.begin(), abs_alpha.end());
//                    double tmp = pow(-1, min_idx) * abs_alpha[min_idx];
//                    DMetric += tmp;
                    delete[] HD;
                    continue;
                }
                int temp = 1 << (n - depth);
                auto * pa = (double*)alpha.data() + (depth*N+temp*node);
                auto * pb = (double*)alpha.data() + (depth*N+temp*node+temp/2);
                node *= 2;
                depth++;
                temp /= 2;
                auto *pllr = (double*)alpha.data() + (depth*N+temp*node);
                f(pllr, pa, pb, temp);
                node_state[node_posi] = 1;

            } else if (node_state[node_posi] == 1){
                int temp = 1 << (n - depth);
                auto * pa = (double*)alpha.data() + (depth*N+temp*node);
                auto * pb = (double*)alpha.data() + (depth*N+temp*node+temp/2);
                int ltemp = temp/2;
                int lnode = 2 * node;
                int cdepth = depth + 1;
                auto *pucapl = (u_int8_t*)ucap.data() + (cdepth*N + lnode*ltemp);
                node = 2 * node + 1;
                depth++;
                temp /= 2;
                auto* pllr = (double*)alpha.data() + (depth*N + node*temp);
                g(pllr, pa, pb, pucapl, temp);
                node_state[node_posi] = 2;

            } else{
                int temp = 1 << (n - depth);
                int ctemp = temp / 2;
                int lnode = 2 * node;
                int rnode = 2 * node + 1;
                int cdepth = depth + 1;
                auto *pucapl = (u_int8_t*)ucap.data() + (cdepth*N+ctemp*lnode);
                auto* pucapr = (u_int8_t*)ucap.data() + (cdepth*N+ctemp*rnode);
                auto* pucap = (u_int8_t*)ucap.data() + (depth*N+temp*node);
                u(pucap, pucapl, pucapr, ctemp);
                if (node == 0 and depth == 0){
                    done = true;
                } else{
                    node /= 2;
                    depth -= 1;
                }
                node_state[node_posi] = 3;
            }
        }
    }
    return DMetric;
}