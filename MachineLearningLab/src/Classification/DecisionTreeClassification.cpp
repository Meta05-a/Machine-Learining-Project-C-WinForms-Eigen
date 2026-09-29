#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/Node.h"

using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	for (std::vector<double>& x : X) {
		predictions.push_back(traverseTree(x, root));
	}
	return predictions;
}



Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	// Stopping criteria
	if (depth >= max_depth || y.size() < min_samples_split || EntropyFunctions::entropy(y) == 0) {
		return new Node(mostCommonLabel(y), true); // leaf node
	}

	int n_samples = X.size();
	int n_features = X[0].size();
	std::vector<int> features(n_features);
	std::iota(features.begin(), features.end(), 0);
	std::random_shuffle(features.begin(), features.end()); // random feature selection for splitting

	double best_gain = -1.0;
	int split_idx = -1;
	double split_thresh = 0.0;

	// Loop through candidate features and potential split thresholds
	for (int feature : features) {
		std::vector<double> X_column(n_samples);
		for (int i = 0; i < n_samples; ++i) {
			X_column[i] = X[i][feature];
		}

		std::set<double> thresholds(X_column.begin(), X_column.end());
		for (double threshold : thresholds) {
			double gain = informationGain(y, X_column, threshold);
			if (gain > best_gain) {
				best_gain = gain;
				split_idx = feature;
				split_thresh = threshold;
			}
		}
	}

	if (best_gain == -1.0) {
		return new Node(mostCommonLabel(y), true); // Leaf node if no split
	}

	// Split the data
	std::vector<std::vector<double>> X_left, X_right;
	std::vector<double> y_left, y_right;
	for (int i = 0; i < n_samples; ++i) {
		if (X[i][split_idx] <= split_thresh) {
			X_left.push_back(X[i]);
			y_left.push_back(y[i]);
		}
		else {
			X_right.push_back(X[i]);
			y_right.push_back(y[i]);
		}
	}

	Node* left = growTree(X_left, y_left, depth + 1);
	Node* right = growTree(X_right, y_right, depth + 1);

	return new Node(split_idx, split_thresh, left, right); // Internal node
}



/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	std::vector<double> left_y, right_y;
	for (size_t i = 0; i < X_column.size(); ++i) {
		if (X_column[i] <= split_thresh) {
			left_y.push_back(y[i]);
		}
		else {
			right_y.push_back(y[i]);
		}
	}

	if (left_y.empty() || right_y.empty()) return 0;

	double parent_entropy = EntropyFunctions::entropy(y);
	double left_weight = static_cast<double>(left_y.size()) / y.size();
	double right_weight = static_cast<double>(right_y.size()) / y.size();

	double child_entropy = left_weight * EntropyFunctions::entropy(left_y) + right_weight * EntropyFunctions::entropy(right_y);
	return parent_entropy - child_entropy;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonLabel(std::vector<double>& y) {
	std::unordered_map<double, int> label_count;
	for (double label : y) {
		label_count[label]++;
	}

	return std::max_element(label_count.begin(), label_count.end(),
		[](const std::pair<double, int>& a, const std::pair<double, int>& b) {
			return a.second < b.second;
		})->first;
}



// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {
	if (node->isLeafNode()) {
		return node->value;
	}

	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	else {
		return traverseTree(x, node->right);
	}
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and
/// then returns a tuple containing the evaluation metrics for the training and test sets,
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
	DataPreprocessor DataPreprocessor;
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

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

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
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}