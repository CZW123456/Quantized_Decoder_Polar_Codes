//
// Created by Zhiwei Cao on 2020/5/14.
//

#include "IBQuantizers.h"
#include <pybind11/stl.h>
#include <vector>

using namespace std;
using namespace cv;

template<typename T> inline std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});

    return array_index;
}

double inline log2_stable(double value){
    if (value <= 0){
        return 1e-6;
    } else{
        return log2(value);
    }
}

double inline mutual_information(Mat &pdf){
    int X = pdf.rows;
    int Y = pdf.cols;
//    printf("X = %d, Y = %d\n", X, Y);
    Mat px = Mat::zeros(1, X, CV_64F);
    Mat py = Mat::zeros(1, Y, CV_64F);
//    printf("compute px and py\n");
    for (int x = 0; x < X; ++x) {
        for (int y = 0; y < Y; ++y) {
            px.at<double>(x) += pdf.at<double>(x, y);
            py.at<double>(y) += pdf.at<double>(x, y);
        }
    }
//    print(px);
//    print(py);
//    printf("compute I\n");
    double MI = 0.0;
    for (int x = 0; x < X; ++x) {
        for (int y = 0; y < Y; ++y) {
            MI += pdf.at<double>(x, y) * log2_stable(pdf.at<double>(x, y) / (px.at<double>(0, x) * py.at<double>(0, y) + 1e-31));
        }
    }
    return MI;
}

ModifiedsIBQuantizer::ModifiedsIBQuantizer(double beta_, int nrun_) {
    beta = beta_;
    nrun = nrun_;
}

vector<double> ModifiedsIBQuantizer::compute_merge_cost(Mat &p_t, Mat &p_x_given_t, int border_between_cluster) {
    int bbc = border_between_cluster;
    int K = p_t.cols - 1;
    vector<double> p_t_bar = {p_t.at<double>(K) + p_t.at<double>(bbc), p_t.at<double>(K) + p_t.at<double>(bbc + 1)};
    vector<double> pi1 = {p_t.at<double>(K)/p_t_bar[0], p_t.at<double>(K)/p_t_bar[1]};
    vector<double> pi2 = {p_t.at<double>(bbc)/p_t_bar[0], p_t.at<double>(bbc+1)/p_t_bar[1]};
    vector<double> pdf1 = {p_x_given_t.at<double>(0, K), p_x_given_t.at<double>(1, K)};
    vector<vector<double>> pdf2(2);
    pdf2[0].resize(2), pdf2[1].resize(2);
    pdf2[0][0] = p_x_given_t.at<double>(0, bbc);
    pdf2[0][1] = p_x_given_t.at<double>(1, bbc);
    pdf2[1][0] = p_x_given_t.at<double>(0, bbc+1);
    pdf2[1][1] = p_x_given_t.at<double>(1, bbc+1);
    vector<vector<double>> p_tilde(2);
    p_tilde[0].resize(2), p_tilde[1].resize(2);
    p_tilde[0][0] = pi1[0]*pdf1[0]+pi2[0]*pdf2[0][0];
    p_tilde[0][1] = pi1[0]*pdf1[1]+pi2[0]*pdf2[0][1];
    p_tilde[1][0] = pi1[1]*pdf1[0]+pi2[1]*pdf2[1][0];
    p_tilde[1][1] = pi1[1]*pdf1[1]+pi2[1]*pdf2[1][1];
    vector<double> kl_divergence1(2);
    vector<double> kl_divergence2(2);
    kl_divergence1[0] = pdf1[0] * log2_stable(pdf1[0]/(p_tilde[0][0]))+pdf1[1] * log2_stable(pdf1[1]/(p_tilde[0][1]));
    kl_divergence1[1] = pdf1[0] * log2_stable(pdf1[0]/(p_tilde[1][0]))+pdf1[1] * log2_stable(pdf1[1]/(p_tilde[1][1]));
    kl_divergence2[0] = pdf2[0][0] * log2_stable(pdf2[0][0]/(p_tilde[0][0])) + pdf2[0][1] * log2_stable(pdf2[0][1]/(p_tilde[0][1]));
    kl_divergence2[1] = pdf2[1][0] * log2_stable(pdf2[1][0]/(p_tilde[1][0])) + pdf2[1][1] * log2_stable(pdf2[1][1]/(p_tilde[1][1]));
    vector<double> merge_cost = {p_t_bar[0] * (pi1[0]*kl_divergence1[0]+pi2[0]*kl_divergence2[0]),
                                 p_t_bar[1] * (pi1[1]*kl_divergence1[1]+pi2[1]*kl_divergence2[1])};
    return merge_cost;
}

py::tuple ModifiedsIBQuantizer::find_quantizer(py::array_t<double> &p_y_given_x_, py::array_t<int> &border_vector_, int K) {

    // prepare input variables
    py::buffer_info buf_p_y_given_x_ = p_y_given_x_.request();
    Mat p_y_given_x(buf_p_y_given_x_.shape[0], buf_p_y_given_x_.shape[1], CV_64F, (double_t*)buf_p_y_given_x_.ptr);
    py::buffer_info buf_border_vector = border_vector_.request();
    Mat border_vector(buf_border_vector.shape[0], buf_border_vector.shape[1], CV_32S, (int*)buf_border_vector.ptr);

    // algorithm begins

    // sort symbols in Y according to their LLRs
    int M = p_y_given_x.cols;
    vector<double> llr(M, 0);
    for (int i = 0; i < M; ++i) {
        llr[i] = log2_stable(p_y_given_x.at<double>(0, i) / p_y_given_x.at<double>(1, i));
    }
    vector<int> permutation = argsort<double>(llr);
    Mat tmp(p_y_given_x.size(), CV_64F);
    for (int i = 0; i < M; ++i) {
        tmp.at<double>(0, i) = p_y_given_x.at<double>(0, permutation[i]);
        tmp.at<double>(1, i) = p_y_given_x.at<double>(1, permutation[i]);
    }
    tmp.copyTo(p_y_given_x);

    // compute p(y) and p(x)
    vector<double> px = {0.5,  0.5};
    vector<double> py(M, 0);
    for (int i = 0; i < M; ++i) {
        py[i] = px[0]*p_y_given_x.at<double>(0, i) + px[1]*p_y_given_x.at<double>(1, i);
    }
    // preallocate arrays
    vector<double> I_TX(nrun);
    vector<vector<int>> p_t_given_y_mats(nrun);
    // run for-loop for each number of run
    for (int run = 0; run < nrun; ++run) {
        // initialize LUTs with random border vector
        vector<int> p_t_given_y(M, 0);
        vector<int> num_element_per_cluster(K, 0);
        int a = 0;
        for (int i = 0; i < K; ++i) {
            for (int j = a; j < border_vector.at<int>(run, i); ++j) {
                p_t_given_y[j] = i;
            }
            num_element_per_cluster[i] = border_vector.at<int>(run, i) - a;
            a = border_vector.at<int>(i);
        }
        // Processing
        bool is_change = true;
        int max_torelence = 50;
        int cnt = 0;
        while (is_change and cnt < max_torelence){
            is_change = false;
            for (int border_between_cluster = 0; border_between_cluster < K - 1; ++border_between_cluster) {
                bool done_left_to_right = false;
                bool done_right_to_left = false;
                // check the last element
                while (!done_left_to_right){
                    done_left_to_right = true;
                    int old_cluster = border_between_cluster;
                    int last_element = M - 1;
                    for (int i = M - 1; i >= 0; --i) {
                        if (p_t_given_y[i] == border_between_cluster){
                            last_element = i;
                            break;
                        }
                    }
                    // if old cluster is empty
                    if (num_element_per_cluster[old_cluster] > 1){
                        p_t_given_y[last_element] = -1;
                        Mat p_t = Mat::zeros(1, K + 1, CV_64F);
                        Mat p_x_and_t = Mat::zeros(2, K + 1, CV_64F);
                        Mat p_x_given_t = Mat::zeros(2, K + 1, CV_64F);
                        for (int i = 0; i < M; ++i) {
                            int symbol = p_t_given_y[i];
                            if (symbol != -1) {
                                p_t.at<double>(symbol) += py[i];
                                p_x_and_t.at<double>(0, symbol) += px[0] * p_y_given_x.at<double>(0, i);
                                p_x_and_t.at<double>(1, symbol) += px[1] * p_y_given_x.at<double>(1, i);
                            }
                        }
                        p_t.at<double>(K) = py[last_element];
                        p_x_and_t.at<double>(0, K) = px[0] * p_y_given_x.at<double>(0, last_element);
                        p_x_and_t.at<double>(1, K) = px[1] * p_y_given_x.at<double>(1, last_element);
                        for (int i = 0; i < K + 1; ++i) {
                            p_x_given_t.at<double>(0, i) = p_x_and_t.at<double>(0, i) / p_t.at<double>(i);
                            p_x_given_t.at<double>(1, i) = p_x_and_t.at<double>(1, i) / p_t.at<double>(i);
                        }
                        vector<double> merge_costs = compute_merge_cost(p_t, p_x_given_t, border_between_cluster);
                        if (merge_costs[0] < merge_costs[1]){
                            p_t_given_y[last_element] = border_between_cluster;
                        } else{
                            p_t_given_y[last_element] = border_between_cluster + 1;
                            is_change = true;
                            done_left_to_right = false;
                            num_element_per_cluster[border_between_cluster] -= 1;
                            num_element_per_cluster[border_between_cluster+1] += 1;
                        }
                    }
                }
                // check the first order
                while (!done_right_to_left){
                    done_right_to_left = true;
                    int old_cluster = border_between_cluster + 1;
                    int first_element = 0;
                    for (int i = 0; i < M; ++i) {
                        if (p_t_given_y[i] == old_cluster){
                            first_element = i;
                            break;
                        }
                    }
                    // if old cluster is empty
                    if (num_element_per_cluster[old_cluster] > 1){
                        p_t_given_y[first_element] = -1;
                        Mat p_t = Mat::zeros(1, K + 1, CV_64F);
                        Mat p_x_and_t = Mat::zeros(2, K + 1, CV_64F);
                        Mat p_x_given_t = Mat::zeros(2, K + 1, CV_64F);
                        for (int i = 0; i < M; ++i) {
                            int symbol = p_t_given_y[i];
                            if (symbol != -1) {
                                p_t.at<double>(symbol) += py[i];
                                p_x_and_t.at<double>(0, symbol) += px[0] * p_y_given_x.at<double>(0, i);
                                p_x_and_t.at<double>(1, symbol) += px[1] * p_y_given_x.at<double>(1, i);
                            }
                        }
                        p_t.at<double>(K) = py[first_element];
                        p_x_and_t.at<double>(0, K) = px[0] * p_y_given_x.at<double>(0, first_element);
                        p_x_and_t.at<double>(1, K) = px[1] * p_y_given_x.at<double>(1, first_element);
                        for (int i = 0; i < K + 1; ++i) {
                            p_x_given_t.at<double>(0, i) = p_x_and_t.at<double>(0, i) / p_t.at<double>(i);
                            p_x_given_t.at<double>(1, i) = p_x_and_t.at<double>(1, i) / p_t.at<double>(i);
                        }
                        vector<double> merge_costs = compute_merge_cost(p_t, p_x_given_t, border_between_cluster);
                        if (merge_costs[0] < merge_costs[1]){
                            p_t_given_y[first_element] = border_between_cluster;
                            is_change = true;
                            done_right_to_left = false;
                            num_element_per_cluster[border_between_cluster] += 1;
                            num_element_per_cluster[old_cluster] -= 1;
                        } else{
                            p_t_given_y[first_element] = old_cluster;
                        }
                    }
                }
            }
            cnt++;
        }
        // finish current run, compute p(t), p(x|t), I_XY, I_XT given the obtained LUT p(t|y)
        Mat p_x_and_t = Mat::zeros(2, K, CV_64F);
        for (int i = 0; i < M; ++i) {
            int symbol = p_t_given_y[i];
            p_x_and_t.at<double>(0, symbol) += px[0] * p_y_given_x.at<double>(0, i);
            p_x_and_t.at<double>(1, symbol) += px[1] * p_y_given_x.at<double>(1, i);
        }
        I_TX[run] = mutual_information(p_x_and_t);
        p_t_given_y_mats[run] = p_t_given_y;
    }
    int winner = std::distance(I_TX.begin(), max_element(I_TX.begin(), I_TX.end()));
    vector<int> p_t_given_y = p_t_given_y_mats[winner];
    auto p_t_given_x = py::array_t<double>({2, K});
    py::buffer_info buf_p_t_given_x = p_t_given_x.request();
    auto* ptr_p_t_given_x = (double*)buf_p_t_given_x.ptr;
    memset(ptr_p_t_given_x, 0, sizeof(double)*2*K);
    for (int i = 0; i < M; ++i) {
        int symbol = p_t_given_y[i];
        ptr_p_t_given_x[symbol] += p_y_given_x.at<double>(0, i);
        ptr_p_t_given_x[symbol+K] += p_y_given_x.at<double>(1, i);
    }
    return py::make_tuple(p_t_given_y, p_t_given_x, permutation, I_TX[winner]);
}