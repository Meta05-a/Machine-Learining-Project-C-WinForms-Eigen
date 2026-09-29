#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
    int total_samples = y.size();
    std::vector<double> hist;
    std::unordered_map<double, int> label_map;
    double entropy = 0.0;

    // Convert labels to unique integers and count their occurrences
    for (const double& label : y) {
        label_map[label]++;
    }

    for (const auto& pair : label_map) {
        hist.push_back(pair.second);
    }

    // Compute the probability and entropy
    for (int count : hist) {
        double p = static_cast<double>(count) / total_samples;
        entropy -= p * log2(p);
    }

    return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
    std::vector<double> hist;
    std::unordered_map<double, int> label_map;
    int total_samples = idxs.size();
    double entropy = 0.0;

    // Convert labels to unique integers and count their occurrences
    for (int idx : idxs) {
        label_map[y[idx]]++;
    }

    for (const auto& pair : label_map) {
        hist.push_back(pair.second);
    }

    // Compute the probability and entropy
    for (int count : hist) {
        double p = static_cast<double>(count) / total_samples;
        entropy -= p * log2(p);
    }

    return entropy;
}


