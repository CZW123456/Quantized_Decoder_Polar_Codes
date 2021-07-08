//
// Created by Zhiwei Cao on 2020/5/8.
//

#include <pybind11/stl.h>
#include <algorithm>
#include "LLRQuantizer.h"

template<typename T> inline std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}

double LLROptLSQuantizer::compute_partial_quantization_noise(Mat &density, Mat &quanta) {
    CV_Assert(density.size() == quanta.size());
    CV_Assert(density.rows == 1);
    int l = density.cols;
    double sum_quanta = 0.0;
    double sum_density = 0.0;
    double *pdensity = density.ptr<double>(0);
    double *pquanta = quanta.ptr<double>(0);
    // determine new quanta
    for (int i = 0; i < l; ++i) {
        sum_quanta += pdensity[i] * pquanta[i];
        sum_density += pdensity[i];
    }
    // compute the quantization noise given the new quanta
    double new_quanta = sum_quanta / sum_density;
    double quantization_noise = 0.0;
    for (int i = 0; i < l; ++i) {
        quantization_noise += (pquanta[i] - new_quanta) * (pquanta[i] - new_quanta) * pdensity[i];
    }
    return quantization_noise;
}

Mat LLROptLSQuantizer::precompute_quantization_noise_table(Mat& density, Mat& quantas, int M, int K) {
    Mat table(M, M+1, CV_64F);
    table.setTo(0);
    auto *pllr_density = density.ptr<double>(0);
    auto *pllr_quanta = quantas.ptr<double>(0);
    for (int a_prime = 0; a_prime < M; ++a_prime) {
        int max_a = min(a_prime + M - K + 1, M);
        for (int a = a_prime + 1; a < max_a + 1; ++a){
            int len = a - a_prime;
            Mat tmp_llr_density(1, len, CV_64F), tmp_llr_quanta(1, len, CV_64F);
            memcpy(tmp_llr_density.ptr<double>(0), pllr_density + a_prime, len * sizeof(double_t));
            memcpy(tmp_llr_quanta.ptr<double>(0), pllr_quanta + a_prime, len * sizeof(double_t));
            table.at<double_t>(a_prime, a) = compute_partial_quantization_noise(tmp_llr_density, tmp_llr_quanta);
        }
    }
    return table;
}

py::tuple LLROptLSQuantizer::find_OptLS_quantizer(py::array_t<double> &llr_density, py::array_t<double> &llr_quanta, int M, int K) {
    py::buffer_info buf_density = llr_density.request();
    Mat density(1, buf_density.shape[0], CV_64F);
    py::buffer_info buf_quantas = llr_quanta.request();
    Mat quantas_before_sorting(1, buf_quantas.shape[0], CV_64F, (double*)buf_quantas.ptr);
    Mat quantas(1, buf_quantas.shape[0], CV_64F);
    CV_Assert(M >= K);
    vector<int> permutation = argsort<double>(quantas_before_sorting);
    auto* pbuf_density = (double*)buf_density.ptr;
    auto* pbuf_quanta_before_sorting = (double*)buf_quantas.ptr;
    for (int i = 0; i < M; ++i) {
        density.at<double>(0, i) = pbuf_density[permutation[i]];
        quantas.at<double>(0, i) = pbuf_quanta_before_sorting[permutation[i]];
    }

    Mat table = precompute_quantization_noise_table(density, quantas, M, K);
//    print(table);
//    exit(0);
    vector<int> Az(K + 1, 0);
    Az[K] = M;
    Mat state_table = Mat::zeros(M - K + 1, K + 1, CV_64F);
    Mat local_min = Mat::ones(state_table.size(), CV_32S) * (1E30);
    for (int i = 0; i < state_table.rows; ++i) {
        state_table.at<double>(i, 1) = table.at<double>(0, 1 + i);
        local_min.at<int>(i, 1) = 0;
    }

    // dynamic programming begin
    // forward computing
    for (int z = 2; z < K + 1; ++z) {
        if (z < K){
            for (int a = z; a < z + M - K + 1; ++a) {
                int a_idx = a - z;
                int a_prime_begin = z - 1;
                int a_prime_end = a - 1;
                vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
                vector<int> tmp_idx;
                for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                    tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                    tmp_idx.push_back(a_prime);
                }
                int min_tmp_idx = argmin(tmp.begin(), tmp.end());
                local_min.at<int>(a_idx, z) = tmp_idx[min_tmp_idx];
                state_table.at<double>(a_idx, z) = tmp[min_tmp_idx];
            }
        } else{
            int a = M;
            int a_prime_begin = z - 1;
            int a_prime_end = a - 1;
            vector<double> tmp(a_prime_end - a_prime_begin + 1, 0);
            vector<int> tmp_idx;
            for (int a_prime = a_prime_begin, cnt = 0; a_prime < a_prime_end + 1; ++a_prime, ++cnt) {
                tmp[cnt] = state_table.at<double>(a_prime - a_prime_begin, z - 1) + table.at<double>(a_prime, a);
                tmp_idx.push_back(a_prime);
            }
            int min_tmp_idx = argmin(tmp.begin(), tmp.end());
            local_min.at<int>(M - K, z) = tmp_idx[min_tmp_idx];
            state_table.at<double>(M - K, z) = tmp[min_tmp_idx];
        }
    }
    // backward tracing
    Az[K - 1] = local_min.at<int>(M - K, K);
    int opt_idx = Az[K - 1];
    for (int z = K - 1; z > 1; --z) {
        opt_idx = local_min.at<int>(opt_idx - z, z);
        Az[z - 1] = opt_idx;
    }

    // compressed density, quanta and LUT
    auto compressed_density = py::array_t<double>({1, K});
    py::buffer_info buf_compressed_density = compressed_density.request();
    auto* ptr_compressed_density = (double*)buf_compressed_density.ptr;
    memset(ptr_compressed_density, 0, sizeof(int)*K);

    auto compressed_quantas = py::array_t<double>({1, K});
    py::buffer_info buf_compressed_quantas = compressed_quantas.request();
    auto* ptr_compressed_quantas = (double*)buf_compressed_quantas.ptr;
    memset(ptr_compressed_density, 0, sizeof(int)*K);

    auto lut = py::array_t<int>({1, M});
    py::buffer_info buf_lut = lut.request();
    auto* ptr_lut = (int*)buf_lut.ptr;
    memset(ptr_lut, 0, sizeof(int)*M);

    for (int i = 0; i < K; ++i) {
        int begin = Az[i];
        int end = Az[i + 1];
        double tmp_quanta = 0.;
        double tmp_density = 0.;
        for (int j = begin; j < end; ++j) {
            ptr_lut[permutation[j]] = i;
            tmp_density += density.at<double>(0, permutation[j]);
            tmp_quanta += density.at<double>(0, permutation[j]) * quantas.at<double>(0, permutation[j]);
        }
        ptr_compressed_density[i] = tmp_density;
        ptr_compressed_quantas[i] = tmp_quanta / tmp_density;
    }
    return py::make_tuple(compressed_density, compressed_quantas, lut, state_table.at<double>(M - K, K));
}