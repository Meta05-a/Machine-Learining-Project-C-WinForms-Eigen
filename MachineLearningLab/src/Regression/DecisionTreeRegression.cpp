#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <chrono>
#include <unordered_set>
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
    : min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
    n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
    root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;

    for (auto v : X) {
        predictions.push_back(traverseTree(v, root));
    }

    return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
    int split_idx = -1;
    double split_thresh = 0.0;
    double best_MSE = std::numeric_limits<double>::infinity();


    // Stopping criteria
    if (depth >= max_depth || X.size() < min_samples_split) {
        return new Node(-1, 0.0, nullptr, nullptr, mean(y)); // Create a leaf node with the most common label
    }


    // Iterate over all features
    for (size_t feature_idx = 0; feature_idx < X[0].size(); ++feature_idx) {
        // Get the column of feature values
        std::vector<double> feature_column(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            feature_column[i] = X[i][feature_idx];
        }

        // Find unique threshold values to split on
        std::set<double> unique_thresholds(feature_column.begin(), feature_column.end());
         
        // Test all thresholds for this feature
        for (const auto& threshold : unique_thresholds) {
            double MSE = meanSquaredError(y, feature_column, threshold);

            // If this split is the best so far, save it
            if (MSE < best_MSE) {
                best_MSE = MSE;
                split_idx = feature_idx;
                split_thresh = threshold;
            }
        }
    }

    // If no valid split was found, return a leaf node
    if (best_MSE == std::numeric_limits<double>::infinity()) {
        return new Node(-1, 0.0, nullptr, nullptr, mean(y)); // Leaf node with the most common label
    }

    // Split the data into left and right subsets
    std::vector<std::vector<double>> left_X, right_X;
    std::vector<double> left_y, right_y;
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][split_idx] <= split_thresh) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        }
        else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }

    // Grow the left and right subtrees recursively
    Node* left = growTree(left_X, left_y, depth + 1);
    Node* right = growTree(right_X, right_y, depth + 1);

    // Return a new node with the split threshold and child nodes
    return new Node(split_idx, split_thresh, left, right);
}



/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
    double mse = 0.0;
    double mse_l = 0.0;
    double mse_r = 0.0; 

    // Initialize variables for the two subsets
    std::vector<double> left_y;
    std::vector<double> right_y;

    // Split the dataset based on the threshold
    for (size_t i = 0; i < X_column.size(); ++i) {
        if (X_column[i] <= split_thresh) {
            left_y.push_back(y[i]);
        }
        else {
            right_y.push_back(y[i]);
        }
    }

    // If there's no valid split, return MSE as infinity
    if (left_y.empty() || right_y.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    // Calculate the mean for each subset
    double left_mean = mean(left_y);
    double right_mean = mean(right_y);
    double w_l = left_y.size()/y.size();
    double w_r = right_y.size()/y.size();
    

   

    // Calculate the MSE for each subset
    for (const auto& val : left_y) {
        mse_l += std::pow(val - left_mean, 2); // squared error for left subset
    }

    
    
    for (const auto& val : right_y) {
        mse_r += std::pow(val - right_mean, 2); // squared error for right subset
    }

    
    

    //mse = (w_l * mse_l + w_r * mse_r);
    //  Return the average MSE
    mse = (mse_l + mse_r)  / y.size(); // Average over total number of samples


    return mse;
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

    double meanValue = 0.0;

    // calculate the mean
    for (auto& v : values) {
        meanValue += v;
    }

    meanValue = meanValue / values.size();

    return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

    if (node->isLeafNode()) {
        return node->value;
    }

    if (x[node->feature] <= node->threshold) {
        return traverseTree(x, node->left);
    }

    return traverseTree(x, node->right);
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and
/// then returns a tuple containing the evaluation metrics for the training and test sets,
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio, double& elapsed_time_ms) {
    try {
        // Check if the file path is empty
        if (filePath.empty()) {
            MessageBox::Show("Please browse and select the dataset file from your PC.");
            return {}; // Return an empty vector since there's no valid file path
        }

        // Attempt to open the file
        std::ifstream file(filePath);
        if (!file.is_open()) {
            MessageBox::Show("Failed to open the dataset file");
            return {}; // Return an empty vector since file couldn't be opened
        }
        // Load the dataset from the file path
        std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

        // Convert the dataset from strings to doubles
        std::vector<std::vector<double>> dataset;
        bool isFirstRow = true; // Flag to identify the first row

        for (const auto& row : data) {
            if (isFirstRow) {
                isFirstRow = false;
                continue; // Skip the first row (header)
            }

            std::vector<double> convertedRow;
            for (const auto& cell : row) {
                try {
                    double value = std::stod(cell);
                    convertedRow.push_back(value);
                }
                catch (const std::exception& e) {
                    // Handle the exception or set a default value
                    std::cerr << "Error converting value: " << cell << std::endl;
                    // You can choose to set a default value or handle the error as needed
                }
            }
            dataset.push_back(convertedRow);
        }

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Time the actual machine learning algorithm execution (fit + predict)
        auto ml_t0 = std::chrono::high_resolution_clock::now();

        // Fit the model to the training data
        fit(trainData, trainLabels);

        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);

        // Make predictions on the training data
        std::vector<double> trainPredictions = predict(trainData);

        auto ml_t1 = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(ml_t1 - ml_t0).count();

        // Calculate evaluation metrics (e.g., MAE, MSE)
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        // Calculate evaluation metrics for training data
        double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
        double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
        double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

        return std::make_tuple(test_mae, test_rmse, test_rsquared,
            train_mae, train_rmse, train_rsquared,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            std::vector<double>(), std::vector<double>(),
            std::vector<double>(), std::vector<double>());
    }
}