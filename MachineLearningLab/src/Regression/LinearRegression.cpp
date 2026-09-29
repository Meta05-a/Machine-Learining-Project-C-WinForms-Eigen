#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///


// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("Size of trainData and trainLabels must match");
    }

    int numSamples = trainData.size();
    int numFeatures = trainData[0].size();

    // Create design matrix X (with an additional column for the intercept)
    Eigen::MatrixXd X(numSamples, numFeatures + 1);
    Eigen::VectorXd y(numSamples);

    for (int i = 0; i < numSamples; ++i) {
        y(i) = trainLabels[i];
        X(i, 0) = 1.0; // Intercept term
        for (int j = 0; j < numFeatures; ++j) {
            X(i, j + 1) = trainData[i][j];
        }
    }

    // Calculate coefficients using the least squares method: (X^T * X)^-1 * X^T * y
    m_coefficients = (X.transpose() * X).inverse() * X.transpose() * y;
}


void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels,  Eigen::VectorXd& coefficients) {
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("Size of trainData and trainLabels must match");
    }

    int numSamples = trainLabels.size();
    int numFeatures = trainData[0].size();

    
        
    coefficients = Eigen::VectorXd::Random(numFeatures + 1); // Initialize coefficients

    // Create design matrix X (with an additional column for the intercept)
    Eigen::MatrixXd X(numSamples, numFeatures +1);
    Eigen::VectorXd y(numSamples);
    
    for (int i = 0; i < numSamples; ++i) {
        y(i) = trainLabels[i];
        X(i, 0) = 1.0; // Intercept term
        for (int j = 0; j < numFeatures; ++j) {
            X(i, j +1) = trainData[i][j];
        }
    }

    

    learning_rate = 0.01;
    epochs = 1000; 
    
    Eigen::VectorXd  predictions;
    Eigen::VectorXd  errors;
    Eigen::VectorXd  gradient;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        
        predictions = X * coefficients; 
        errors = predictions - y; 
        gradient = X.transpose() * errors;
        coefficients -= (learning_rate/numSamples) * gradient; 
    }
       
}

// Matrix Form predict function
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {
    if (m_coefficients.size() == 0) {
        throw std::runtime_error("Model has not been fitted yet");
    }

    int numSamples = testData.size();
    int numFeatures = testData[0].size();

    // Create design matrix X for test data
    Eigen::MatrixXd X(numSamples, numFeatures + 1);
    for (int i = 0; i < numSamples; ++i) {
        X(i, 0) = 1.0; // Intercept term
        for (int j = 0; j < numFeatures; ++j) {
            X(i, j + 1) = testData[i][j];
        }
    }

    // Make predictions using the coefficients
    Eigen::VectorXd predictions = X * m_coefficients;

    // Convert predictions to a vector
    std::vector<double> result(predictions.data(), predictions.data() + predictions.size());
    return result;
}

// Overloaded predict function that accepts a vector of coefficients (if needed)
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData, const Eigen::VectorXd& coefficients) {
    if (m_coefficients.size() == 0) {
        throw std::runtime_error("Model has not been fitted yet");
    }

    int numSamples = testData.size();
    int numFeatures = testData[0].size();

    // Create design matrix X for test data
    Eigen::MatrixXd X(numSamples, numFeatures + 1);
    for (int i = 0; i < numSamples; ++i) {
        X(i, 0) = 1.0; // Intercept term
        for (int j = 0; j < numFeatures; ++j) {
            X(i, j + 1) = testData[i][j];
        }
    }

    // Make predictions using the coefficients
    Eigen::VectorXd predictions = X * coefficients;

    // Convert predictions to a vector
    std::vector<double> result(predictions.data(), predictions.data() + predictions.size());
    return result;
}

/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio, double& elapsed_time_ms) {
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

        // Normalize features only (keep labels in original scale)
        DataPreprocessor::normalizeDataset(trainData);
        DataPreprocessor::normalizeDataset(testData);

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