#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"

#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <set>

using namespace System::Windows::Forms;


// ============================================================
// KMeans Constructor
// ============================================================

KMeans::KMeans(int numClusters, int maxIterations)
    : numClusters_(numClusters),
    maxIterations_(maxIterations)
{
}


// ============================================================
// FIT
// Performs K-Means clustering
// ============================================================

void KMeans::fit(const std::vector<std::vector<double>>& data)
{
    // Safety check
    if (data.empty())
        return;

    int numPoints = static_cast<int>(data.size());
    int numFeatures = static_cast<int>(data[0].size());

    if (numPoints < numClusters_)
        return;

    // --------------------------------------------------------
    // Initialize centroids by randomly selecting data points
    // --------------------------------------------------------

    std::set<int> chosenIndices;

    std::mt19937 gen(std::random_device{}());

    std::uniform_int_distribution<> dis(0, numPoints - 1);

    while (chosenIndices.size() < numClusters_)
    {
        int index = dis(gen);
        chosenIndices.insert(index);
    }

    centroids_.resize(
        numClusters_,
        std::vector<double>(numFeatures, 0.0)
    );

    int i = 0;

    for (int index : chosenIndices)
    {
        centroids_[i++] = data[index];
    }


    // --------------------------------------------------------
    // Variables used during training
    // --------------------------------------------------------

    std::vector<int> clusterAssignments(numPoints);

    std::vector<std::vector<double>> newCentroids(
        numClusters_,
        std::vector<double>(numFeatures, 0.0)
    );

    std::vector<int> clusterCounts(numClusters_, 0);


    // ========================================================
    // K-MEANS ITERATIONS
    // ========================================================

    for (int iteration = 0;
        iteration < maxIterations_;
        ++iteration)
    {

        // ----------------------------------------------------
        // Reset values before assigning points
        // ----------------------------------------------------

        std::fill(
            clusterCounts.begin(),
            clusterCounts.end(),
            0
        );

        newCentroids.assign(
            numClusters_,
            std::vector<double>(numFeatures, 0.0)
        );


        // ----------------------------------------------------
        // STEP 1: Assign every point to the closest centroid
        // ----------------------------------------------------

        for (int j = 0; j < numPoints; ++j)
        {
            double minDistance =
                std::numeric_limits<double>::infinity();

            int closestCentroid = 0;


            for (int k = 0; k < numClusters_; ++k)
            {
                double distance = 0.0;

                for (int d = 0; d < numFeatures; ++d)
                {
                    double difference =
                        data[j][d] - centroids_[k][d];

                    distance += difference * difference;
                }

                distance = std::sqrt(distance);


                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestCentroid = k;
                }
            }


            // Save cluster assignment
            clusterAssignments[j] = closestCentroid;

            clusterCounts[closestCentroid]++;


            // Add point to centroid calculation
            for (int d = 0; d < numFeatures; ++d)
            {
                newCentroids[closestCentroid][d]
                    += data[j][d];
            }
        }


        // ----------------------------------------------------
        // STEP 2: Calculate new centroids
        // ----------------------------------------------------

        for (int k = 0; k < numClusters_; ++k)
        {
            if (clusterCounts[k] > 0)
            {
                for (int d = 0; d < numFeatures; ++d)
                {
                    newCentroids[k][d]
                        /= clusterCounts[k];
                }
            }
            else
            {
                // If a cluster has no points,
                // keep its old centroid
                newCentroids[k] = centroids_[k];
            }
        }


        // ----------------------------------------------------
        // STEP 3: Check convergence
        // ALL centroids must have stopped moving
        // ----------------------------------------------------

        bool converged = true;


        for (int k = 0; k < numClusters_; ++k)
        {
            double centroidMovement =
                SimilarityFunctions::euclideanDistance(
                    centroids_[k],
                    newCentroids[k]
                );


            if (centroidMovement >= 1e-4)
            {
                converged = false;
                break;
            }
        }


        // ----------------------------------------------------
        // If all centroids stopped moving, finish
        // ----------------------------------------------------

        if (converged)
        {
            centroids_ = newCentroids;
            break;
        }


        // ----------------------------------------------------
        // Otherwise update centroids and continue
        // ----------------------------------------------------

        centroids_ = std::move(newCentroids);
    }
}


// ============================================================
// PREDICT
// Assigns every point to the closest centroid
// ============================================================

std::vector<int> KMeans::predict(
    const std::vector<std::vector<double>>& data) const
{
    std::vector<int> labels;


    if (data.empty() || centroids_.empty())
        return labels;


    for (size_t j = 0; j < data.size(); ++j)
    {
        double minDistance =
            std::numeric_limits<double>::infinity();

        int closestCentroid = -1;


        for (size_t k = 0;
            k < centroids_.size();
            ++k)
        {
            double distance = 0.0;


            for (size_t d = 0;
                d < data[j].size();
                ++d)
            {
                double difference =
                    data[j][d] - centroids_[k][d];

                distance += difference * difference;
            }


            distance = std::sqrt(distance);


            if (distance < minDistance)
            {
                minDistance = distance;
                closestCentroid =
                    static_cast<int>(k);
            }
        }


        // IMPORTANT:
        // Keep cluster labels as 0, 1, 2
        // instead of changing them to 1, 2, 3.

        labels.push_back(closestCentroid);
    }


    return labels;
}


// ============================================================
// RUN K-MEANS
// ============================================================

std::tuple<
    double,
    double,
    std::vector<int>,
    std::vector<std::vector<double>>
>
KMeans::runKMeans(const std::string& filePath)
{
    DataPreprocessor DataPreprocessor;


    try
    {
        // ----------------------------------------------------
        // Check file path
        // ----------------------------------------------------

        if (filePath.empty())
        {
            MessageBox::Show(
                "Please browse and select the dataset file from your PC."
            );

            return {};
        }


        // ----------------------------------------------------
        // Open dataset
        // ----------------------------------------------------

        std::ifstream file(filePath);


        if (!file.is_open())
        {
            MessageBox::Show(
                "Failed to open the dataset file"
            );

            return {};
        }


        // ----------------------------------------------------
        // Load dataset
        // ----------------------------------------------------

        std::vector<std::vector<double>> dataset;


        DataLoader::loadAndPreprocessDataset(
            filePath,
            dataset
        );


        if (dataset.empty())
        {
            MessageBox::Show(
                "Dataset is empty."
            );

            return {};
        }


        // ----------------------------------------------------
        // Use all data for clustering
        // ----------------------------------------------------

        double trainRatio = 1.0;


        std::vector<std::vector<double>> trainData;

        std::vector<double> trainLabels;

        std::vector<std::vector<double>> testData;

        std::vector<double> testLabels;


        DataPreprocessor::splitDataset(
            dataset,
            trainRatio,
            trainData,
            trainLabels,
            testData,
            testLabels
        );


        if (trainData.empty())
        {
            MessageBox::Show(
                "Training data is empty."
            );

            return {};
        }


        // ----------------------------------------------------
        // TRAIN K-MEANS
        // ----------------------------------------------------

        fit(trainData);


        // ----------------------------------------------------
        // Predict cluster for every training point
        // ----------------------------------------------------

        std::vector<int> labels =
            predict(trainData);


        if (labels.empty())
        {
            MessageBox::Show(
                "No cluster labels were generated."
            );

            return {};
        }


        // ====================================================
        // EVALUATION
        // ====================================================


        // ----------------------------------------------------
        // Davies-Bouldin Index
        // ----------------------------------------------------

        double daviesBouldinIndex =
            Metrics::calculateDaviesBouldinIndex(
                trainData,
                labels
            );


        // ----------------------------------------------------
        // Silhouette Score
        // ----------------------------------------------------

        double silhouetteScore =
            Metrics::calculateSilhouetteScore(
                trainData,
                labels
            );


        // ====================================================
        // PCA
        // ====================================================

        PCADimensionalityReduction pca;


        int num_dimensions = 2;


        std::vector<std::vector<double>> reduced_data =
            pca.performPCA(
                trainData,
                num_dimensions
            );


        // ----------------------------------------------------
        // Return results
        // ----------------------------------------------------

        return std::make_tuple(
            daviesBouldinIndex,
            silhouetteScore,
            std::move(labels),
            std::move(reduced_data)
        );
    }


    catch (const std::exception& e)
    {
        MessageBox::Show(
            "Not Working"
        );


        std::cerr
            << "Exception occurred: "
            << e.what()
            << std::endl;


        return std::make_tuple(
            0.0,
            0.0,
            std::vector<int>(),
            std::vector<std::vector<double>>()
        );
    }
}