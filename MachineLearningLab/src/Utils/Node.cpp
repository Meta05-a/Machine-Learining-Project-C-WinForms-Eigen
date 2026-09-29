#include "Node.h"

                                    // Node class implementation //

// Constructor for internal nodes
Node::Node(int feature, double threshold, Node* left, Node* right, double value)
    : feature(feature), threshold(threshold), left(left), right(right), value(value) {}

// Constructor for leaf nodes
Node::Node(double value, bool isLeaf)
    : feature(-1), threshold(0.0), left(nullptr), right(nullptr), value(value) {}

// Method to check if it's a leaf node
bool Node::isLeafNode() const {
    return left == nullptr && right == nullptr;
}