#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox

// Constructor
LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();  // Count number of unique classes

    // Initialize weights and biases for each class (one-vs-rest)
    std::vector<std::vector<double>> weights(num_classes, std::vector<double>(num_features, 0.0));
    std::vector<double> biases(num_classes, 0.0);

    for (int class_label = 0; class_label < num_classes; ++class_label) {
        std::vector<double> y_binary(y_train.size());

        // Convert problem into binary classification for the current class
        std::transform(y_train.begin(), y_train.end(), y_binary.begin(),
            [class_label](double label) { return label == (class_label + 1) ? 1.0 : 0.0; });

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (int i = 0; i < X_train.size(); ++i) {
                // Add bias term (dot product)
                double weighted_sum = biases[class_label];
                for (int j = 0; j < num_features; ++j) {
                    weighted_sum += weights[class_label][j] * X_train[i][j];
                }

                // Sigmoid function
                double prediction = 1.0 / (1.0 + std::exp(-weighted_sum));

                // Calculate the gradient and update the weights and bias
                double error = prediction - y_binary[i];
                for (int j = 0; j < num_features; ++j) {
                    weights[class_label][j] -= learning_rate * error * X_train[i][j];
                }
                biases[class_label] -= learning_rate * error;  // Update bias
            }
        }

        // Debugging: Print weights and biases for class 3
       /* if (class_label == 2) {  // class_label 2 corresponds to class 3 in the dataset
            std::cout << "Weights for class 3 after training: ";
            for (double w : weights[class_label]) {
                std::cout << w << " ";
            }
            std::cout << "\nBias for class 3: " << biases[class_label] << std::endl;
        }*/
    }

    // Store the final weights and biases
    this->weights = weights;
    this->biases = biases;
}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;
    int num_classes = weights.size();

    for (const auto& instance : X_test) {
        std::vector<double> class_scores(num_classes, 0.0);

        for (int class_label = 0; class_label < num_classes; ++class_label) {
            double weighted_sum = biases[class_label];
            for (int j = 0; j < instance.size(); ++j) {
                weighted_sum += weights[class_label][j] * instance[j];
            }
            class_scores[class_label] = 1.0 / (1.0 + std::exp(-weighted_sum));  // Sigmoid function
        }

        // Predict the class with the highest score
        int predicted_class = std::distance(class_scores.begin(), std::max_element(class_scores.begin(), class_scores.end()));
        predictions.push_back(predicted_class + 1);  // +1 to match original class labels (1, 2, 3)
    }

    return predictions;
}

// runLogisticRegression: Runs the logistic regression and returns evaluation metrics and predictions
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {
    DataPreprocessor DataPreprocessor;
    try {
        if (filePath.empty()) {
            MessageBox::Show("Please browse and select the dataset file from your PC.");
            return {};
        }

        std::ifstream file(filePath);
        if (!file.is_open()) {
            MessageBox::Show("Failed to open the dataset file");
            return {};
        }

        std::vector<std::vector<double>> dataset;
        DataLoader::loadAndPreprocessDataset(filePath, dataset);

        double trainRatio = trainingRatio * 0.01;
        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Fit the model to the training data
        fit(trainData, trainLabels);

        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);

        // Calculate accuracy using the true labels and predicted labels for the test data
        double test_accuracy = Metrics::accuracy(testLabels, testPredictions);

        // Make predictions on the training data
        std::vector<double> trainPredictions = predict(trainData);

        // Calculate accuracy using the true labels and predicted labels for the training data
        double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

        return std::make_tuple(train_accuracy, test_accuracy,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
    }
}
