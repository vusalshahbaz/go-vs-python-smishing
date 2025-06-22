package tree

import (
	"math"
	"sort"
)

// Node represents one decision node or leaf in the tree
type Node struct {
	FeatureIndex int
	Threshold    float64
	Left         *Node
	Right        *Node
	Value        float64 // Predicted class label
	IsLeaf       bool
}

// DecisionTreeClassifier with float64 labels
type DecisionTreeClassifier struct {
	MaxDepth        int
	MinSamplesSplit int
	Root            *Node
}

// Constructor
func NewDecisionTreeClassifier(maxDepth int) *DecisionTreeClassifier {
	return &DecisionTreeClassifier{
		MaxDepth:        maxDepth,
		MinSamplesSplit: 2, // default minimum samples to split
	}
}

// Fit the model
func (dt *DecisionTreeClassifier) Fit(X [][]float64, y []float64) {
	if len(X) == 0 || len(y) == 0 || len(X[0]) == 0 {
		// No data to train
		return
	}
	dt.Root = dt.fitRecursive(X, y, 0)
}

// Predict returns predictions for each sample
func (dt *DecisionTreeClassifier) Predict(X [][]float64) []float64 {
	var preds []float64
	for _, x := range X {
		preds = append(preds, dt.PredictOne(dt.Root, x))
	}
	return preds
}

// Recursive training
func (dt *DecisionTreeClassifier) fitRecursive(X [][]float64, y []float64, depth int) *Node {
	// Stop conditions: max depth, pure node, or not enough samples to split
	if depth >= dt.MaxDepth || gini(y) == 0 || len(y) < dt.MinSamplesSplit {
		return &Node{
			Value:  majorityClass(y),
			IsLeaf: true,
		}
	}

	feature, threshold, found := bestSplit(X, y, dt.MinSamplesSplit)
	if !found {
		// No good split found, make leaf
		return &Node{
			Value:  majorityClass(y),
			IsLeaf: true,
		}
	}

	leftX, rightX, leftY, rightY := split(X, y, feature, threshold)

	// Defensive: If split produces empty partitions, make leaf
	if len(leftY) == 0 || len(rightY) == 0 {
		return &Node{
			Value:  majorityClass(y),
			IsLeaf: true,
		}
	}

	return &Node{
		FeatureIndex: feature,
		Threshold:    threshold,
		Left:         dt.fitRecursive(leftX, leftY, depth+1),
		Right:        dt.fitRecursive(rightX, rightY, depth+1),
		IsLeaf:       false,
	}
}

func (dt *DecisionTreeClassifier) PredictOne(node *Node, x []float64) float64 {
	if node.IsLeaf {
		return node.Value
	}
	if x[node.FeatureIndex] <= node.Threshold {
		return dt.PredictOne(node.Left, x)
	}
	return dt.PredictOne(node.Right, x)
}

// --- Helper functions ---

func gini(y []float64) float64 {
	total := float64(len(y))
	if total == 0 {
		return 0
	}
	count := map[float64]int{}
	for _, label := range y {
		count[label]++
	}
	impurity := 1.0
	for _, c := range count {
		p := float64(c) / total
		impurity -= p * p
	}
	return impurity
}

func majorityClass(y []float64) float64 {
	count := map[float64]int{}
	for _, label := range y {
		count[label]++
	}
	var maxClass float64
	maxCount := -1
	for label, c := range count {
		if c > maxCount {
			maxCount = c
			maxClass = label
		}
	}
	return maxClass
}

func split(X [][]float64, y []float64, featureIndex int, threshold float64) ([][]float64, [][]float64, []float64, []float64) {
	var leftX, rightX [][]float64
	var leftY, rightY []float64

	for i, row := range X {
		if row[featureIndex] <= threshold {
			leftX = append(leftX, row)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, row)
			rightY = append(rightY, y[i])
		}
	}

	return leftX, rightX, leftY, rightY
}

// bestSplit returns the best feature and threshold for a split,
// and a boolean indicating if a valid split was found.
func bestSplit(X [][]float64, y []float64, minSamplesSplit int) (int, float64, bool) {
	if len(X) == 0 || len(X[0]) == 0 {
		return 0, 0, false
	}

	bestGini := math.Inf(1)
	var bestFeature int
	var bestThreshold float64
	foundSplit := false

	nFeatures := len(X[0])

	for feature := 0; feature < nFeatures; feature++ {
		// Extract sorted unique values for the feature
		featureValues := make([]float64, len(X))
		for i := range X {
			featureValues[i] = X[i][feature]
		}
		sort.Float64s(featureValues)

		uniqueVals := uniqueSorted(featureValues)
		// Need at least 2 unique values to split
		if len(uniqueVals) < 2 {
			continue
		}

		// Consider midpoints between unique values as thresholds
		for i := 0; i < len(uniqueVals)-1; i++ {
			threshold := (uniqueVals[i] + uniqueVals[i+1]) / 2

			_, _, leftY, rightY := split(X, y, feature, threshold)

			if len(leftY) < minSamplesSplit || len(rightY) < minSamplesSplit {
				continue // ignore splits that produce too small leaves
			}

			p := float64(len(leftY)) / float64(len(y))
			g := p*gini(leftY) + (1-p)*gini(rightY)

			if g < bestGini {
				bestGini = g
				bestFeature = feature
				bestThreshold = threshold
				foundSplit = true
			}
		}
	}

	return bestFeature, bestThreshold, foundSplit
}

func uniqueSorted(arr []float64) []float64 {
	if len(arr) == 0 {
		return arr
	}
	res := []float64{arr[0]}
	for _, v := range arr[1:] {
		if v != res[len(res)-1] {
			res = append(res, v)
		}
	}
	return res
}
