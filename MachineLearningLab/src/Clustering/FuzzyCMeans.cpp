#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {}


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
	int numDataPoints = data.size();
	initializeMembershipMatrix(numDataPoints);  // Initialize membership matrix
	centroids_.resize(numClusters_, std::vector<double>(data[0].size())); // Resize centroids

	for (int iteration = 0; iteration < maxIterations_; ++iteration) {
		// Update centroids based on current membership matrix
		centroids_ = updateCentroids(data);

		// Update the membership matrix based on the new centroids
		updateMembershipMatrix(data, centroids_);
	}
}



// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));

	for (int i = 0; i < numDataPoints; ++i) {
		double sum = 0.0;
		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] = distribution(generator);
			sum += membershipMatrix_[i][j];
		}
		// Normalize
		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] /= sum;
		}
	}
}



// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>> centroids_) {
	int numDataPoints = data.size();
	for (int i = 0; i < numDataPoints; ++i) {
		double sum = 0.0;
		for (int j = 0; j < numClusters_; ++j) {
			double distance = 0.0;
			for (size_t k = 0; k < data[i].size(); ++k) {
				distance += std::pow(data[i][k] - centroids_[j][k], 2);
			}
			distance = std::sqrt(distance);

			if (distance > 0) {
				membershipMatrix_[i][j] = 1.0 / std::pow(distance, 2 / (fuzziness_ - 1));
			}
			else {
				membershipMatrix_[i][j] = 1.0; // Full membership if the distance is 0
			}
			sum += membershipMatrix_[i][j];
		}
		// Normalize
		for (int j = 0; j < numClusters_; ++j) {
			membershipMatrix_[i][j] /= sum;
		}
	}
}



// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {
	centroids_.assign(numClusters_, std::vector<double>(data[0].size(), 0.0));
	std::vector<double> membershipSum(numClusters_, 0.0);

	for (size_t j = 0; j < numClusters_; ++j) {
		for (size_t i = 0; i < data.size(); ++i) {
			double membership = std::pow(membershipMatrix_[i][j], fuzziness_);
			membershipSum[j] += membership;
			for (size_t k = 0; k < data[i].size(); ++k) {
				centroids_[j][k] += membership * data[i][k];
			}
		}
		// Normalize centroid
		if (membershipSum[j] > 0) {
			for (size_t k = 0; k < centroids_[j].size(); ++k) {
				centroids_[j][k] /= membershipSum[j];
			}
		}
	}
	return centroids_;
}


// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels(data.size());
	for (size_t i = 0; i < data.size(); ++i) {
		double maxMembership = -1.0;
		int clusterIndex = 0;

		for (size_t j = 0; j < numClusters_; ++j) {
			if (membershipMatrix_[i][j] > maxMembership) {
				maxMembership = membershipMatrix_[i][j];
				clusterIndex = j;
			}
		}
		labels[i] = clusterIndex + 1;
	}
	return labels;
}



/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
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

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);


		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}