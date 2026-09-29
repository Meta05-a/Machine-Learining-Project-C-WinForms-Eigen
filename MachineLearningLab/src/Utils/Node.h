#ifndef NODE_H
#define NODE_H
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>


                                // Node class definition //

class Node {
public:
    int feature;
    double threshold;
    double value;  // For leaf nodes, stores the most common label
    Node* left;
    Node* right;

    // Constructor for internal nodes
    Node(int feature, double threshold, Node* left, Node* right, double value = 0.0);

    // Constructor for leaf nodes
    Node(double value, bool isLeaf);

    // Method to check if it's a leaf node
    bool isLeafNode() const;
};

#endif // NODE_H