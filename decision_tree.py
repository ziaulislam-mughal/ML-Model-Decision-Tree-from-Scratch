import numpy as np
import pandas as pd
 
 
# ── 1. ENTROPY ──────────────────────────────────────────────────────────────
 
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))
 
 
# ── 2. INFORMATION GAIN ──────────────────────────────────────────────────────
 
def information_gain(y, x, threshold=None):
    parent_entropy = entropy(y)
    n = len(y)
 
    if threshold is not None:
        # Continuous feature
        left  = y[x <= threshold]
        right = y[x >  threshold]
        if len(left) == 0 or len(right) == 0:
            return 0
        weighted = (len(left)/n)*entropy(left) + (len(right)/n)*entropy(right)
    else:
        # Categorical feature
        weighted = sum(
            (len(y[x == v])/n) * entropy(y[x == v])
            for v in np.unique(x)
        )
 
    return parent_entropy - weighted
 
 
# ── 3. BEST THRESHOLD (for continuous features) ───────────────────────────────
 
def best_threshold(y, x):
    sorted_vals = np.sort(np.unique(x))
    thresholds  = (sorted_vals[:-1] + sorted_vals[1:]) / 2
    gains = [information_gain(y, x, t) for t in thresholds]
    return thresholds[np.argmax(gains)], max(gains)
 
 
# ── 4. NODE ──────────────────────────────────────────────────────────────────
 
class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature   = feature     # column to split on
        self.threshold = threshold   # float (continuous) or None (categorical)
        self.value     = value       # leaf prediction
        self.children  = {}         # {category_value: Node}
        self.left      = None       # continuous: x <= threshold
        self.right     = None       # continuous: x >  threshold
 
 
# ── 5. BUILD TREE ─────────────────────────────────────────────────────────────
 
def majority(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]
 
 
def build_tree(X, y, depth=0, max_depth=10, min_samples=2):
    # Base cases
    if len(np.unique(y)) == 1:
        return Node(value=y[0])
    if depth >= max_depth or len(y) < min_samples:
        return Node(value=majority(y))
 
    # Find best feature
    best_gain, best_feat, best_thresh = -1, None, None
 
    for col in X.columns:
        x = X[col].values
        if X[col].dtype in [np.float64, np.int64]:
            thresh, gain = best_threshold(y, x)
            is_cont = True
        else:
            gain, thresh, is_cont = information_gain(y, x), None, False
 
        if gain > best_gain:
            best_gain, best_feat, best_thresh = gain, col, thresh
 
    if best_gain <= 0:
        return Node(value=majority(y))
 
    # Create node and split
    node = Node(feature=best_feat, threshold=best_thresh)
    x    = X[best_feat].values
 
    if best_thresh is not None:
        # Continuous split
        node.left  = build_tree(X[x <= best_thresh], y[x <= best_thresh], depth+1, max_depth, min_samples)
        node.right = build_tree(X[x >  best_thresh], y[x >  best_thresh], depth+1, max_depth, min_samples)
    else:
        # Categorical split
        for val in np.unique(x):
            mask = x == val
            node.children[val] = build_tree(
                X[mask].drop(columns=[best_feat]), y[mask], depth+1, max_depth, min_samples
            )
 
    return node
 
 
# ── 6. PREDICT ────────────────────────────────────────────────────────────────
 
def predict_one(node, row):
    if node.value is not None:
        return node.value
    if node.threshold is not None:
        child = node.left if row[node.feature] <= node.threshold else node.right
    else:
        child = node.children.get(row[node.feature], next(iter(node.children.values())))
    return predict_one(child, row)
 
 
def predict(tree, X):
    return np.array([predict_one(tree, row) for _, row in X.iterrows()])
 
 
