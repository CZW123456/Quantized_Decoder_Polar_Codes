//
// Created by Zhiwei Cao on 2020/6/11.
//

#include "LLRMIQuantizer.h"

inline double log_stable(double x){
    return x <= 0 ? -1e6 : log(x);
}

template<typename T> inline std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;
    std::sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] <= array[pos2]);});
    return array_index;
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

Mat LLRMIQuantizer::precompute_partial_mutual_information_table(Mat &density, int M, int K) {
    CV_Assert(density.rows == 2);
    Mat table(M, M+1, CV_64F);
    table.setTo(0);
    auto *pllr_density_0 = density.ptr<double>(0);
    auto *pllr_density_1 = density.ptr<double>(1);
    for (int a_prime = 0; a_prime < M; ++a_prime) {
        int max_a = min(a_prime + M - K + 1, M);
        for (int a = a_prime + 1; a < max_a + 1; ++a){
            int len = a - a_prime;
            Mat tmp_llr_density(2, len, CV_64F);
            memcpy(tmp_llr_density.ptr<double>(0), pllr_density_0 + a_prime, len * sizeof(double_t));
            memcpy(tmp_llr_density.ptr<double>(1), pllr_density_1 + a_prime, len * sizeof(double_t));
            table.at<double>(a_prime, a) = compute_partial_mutual_information(tmp_llr_density);
        }
    }
    return table;
}

double LLRMIQuantizer::compute_partial_mutual_information(Mat &density) {
    CV_Assert(density.rows == 2);
    int l = density.cols;
    double sum_p0 = 0;
    double sum_p1 = 0;
    for (int i = 0; i < l; ++i) {
        sum_p0 += density.at<double>(0, i);
        sum_p1 += density.at<double>(1, i);
    }
    double r_i = log(sum_p0 / (sum_p1));
    return 0.5 * ((1 - log2(1 + exp(-r_i))) * sum_p0 + (1 - log2(1 + exp(r_i))) * sum_p1);
}

py::tuple LLRMIQuantizer::find_quantizer_decoder(py::array_t<double> &llr_density, py::array_t<int> &permutation_ndarray, int M, int K) {
    py::buffer_info buf_density = llr_density.request();
    auto* pbuf_density = (double*)buf_density.ptr;
    Mat density(2, buf_density.shape[1], CV_64F);
    py::buffer_info buf_permutation = permutation_ndarray.request();
    auto* pbuf_permutation = (int*)buf_permutation.ptr;
    vector<int> permutation(M);
    memcpy(permutation.data(), pbuf_permutation, sizeof(int)*M);
    CV_Assert(M >= K);
    for (int i = 0; i < M; ++i) {
        density.at<double>(0, i) = pbuf_density[permutation[i]];
        density.at<double>(1, i) = pbuf_density[permutation[i]+M];
    }
    Mat table = precompute_partial_mutual_information_table(density, M, K);
    vector<int> Az(K + 1, 0);
    Az[K] = M;
    Mat state_table = Mat::zeros(M - K + 1, K + 1, CV_64F);
    Mat local_max = Mat::ones(state_table.size(), CV_32S) * (1E30);
    for (int i = 0; i < state_table.rows; ++i) {
        state_table.at<double>(i, 1) = table.at<double>(0, 1 + i);
        local_max.at<int>(i, 1) = 0;
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
                int max_tmp_idx = argmax(tmp.begin(), tmp.end());
                local_max.at<int>(a_idx, z) = tmp_idx[max_tmp_idx];
                state_table.at<double>(a_idx, z) = tmp[max_tmp_idx];
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
            int max_tmp_idx = argmax(tmp.begin(), tmp.end());
            local_max.at<int>(M - K, z) = tmp_idx[max_tmp_idx];
            state_table.at<double>(M - K, z) = tmp[max_tmp_idx];
        }
    }
    // backward tracing
    Az[K - 1] = local_max.at<int>(M - K, K);
    int opt_idx = Az[K - 1];
    for (int z = K - 1; z > 1; --z) {
        opt_idx = local_max.at<int>(opt_idx - z, z);
        Az[z - 1] = opt_idx;
    }
    // compressed density, quanta and LUT
    auto compressed_density = py::array_t<double>({2, K});
    py::buffer_info buf_compressed_density = compressed_density.request();
    auto* ptr_compressed_density = (double*)buf_compressed_density.ptr;

    auto compressed_quantas = py::array_t<double>({1, K});
    py::buffer_info buf_compressed_quantas = compressed_quantas.request();
    auto* ptr_compressed_quantas = (double*)buf_compressed_quantas.ptr;

    auto lut = py::array_t<int>({1, M});
    py::buffer_info buf_lut = lut.request();
    auto* ptr_lut = (int*)buf_lut.ptr;

    auto density_ndarray = py::array_t<double>({2, M});
    py::buffer_info buf_density_ndarray = density_ndarray.request();
    auto* pbuf_density_ndarray = (double*)buf_density_ndarray.ptr;
    for (int i = 0; i < M; ++i) {
        pbuf_density_ndarray[i] = density.at<double>(0, i);
        pbuf_density_ndarray[i+M] = density.at<double>(1, i);
    }

    auto table_ndarray = py::array_t<double>({M, M+1});
    py::buffer_info buf_table_ndarray = table_ndarray.request();
    auto* pbuf_table_ndarray = (double*)buf_table_ndarray.ptr;
    auto* ptable = table.ptr<double>(0);
    for (int i = 0; i < M*(M+1); ++i) {
        pbuf_table_ndarray[i] = ptable[i];
    }
//
//    auto permutation_ndarray = py::array_t<int>({1, M});
//    py::buffer_info buf_permutation_ndarray = permutation_ndarray.request();
//    auto* pbuf_permutation_ndarray = (int*)buf_permutation_ndarray.ptr;
//    for (int i = 0; i < M; ++i) {
//        pbuf_permutation_ndarray[i] = permutation[i];
//    }

    for (int i = 0; i < K; ++i) {
        int begin = Az[i];
        int end = Az[i + 1];
        double sum_p0 = 0.;
        double sum_p1 = 0.;
        for (int j = begin; j < end; ++j) {
            ptr_lut[permutation[j]] = i;
            sum_p0 += density.at<double>(0, j);
            sum_p1 += density.at<double>(1, j);
        }
        ptr_compressed_density[i] = sum_p0;
        ptr_compressed_density[i+K] = sum_p1;
        ptr_compressed_quantas[i] = log_stable(sum_p0 / sum_p1);
    }
    return py::make_tuple(compressed_density, compressed_quantas, lut, state_table.at<double>(M - K, K), density_ndarray, table_ndarray);
}