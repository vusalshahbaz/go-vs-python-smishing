package tree

import (
	"math"
	"math/rand"
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
	MaxFeatures     int // Number of features to consider per split (0 = all, sqrt(n) if negative)
	MaxThresholds   int // Max number of thresholds to test per feature (0 = all)
	Root            *Node
	X               [][]float64 // Store original data
	Y               []float64   // Store original labels
}

// Constructor
func NewDecisionTreeClassifier(maxDepth int) *DecisionTreeClassifier {
	return &DecisionTreeClassifier{
		MaxDepth:        maxDepth,
		MinSamplesSplit: 2, // default minimum samples to split
		MaxFeatures:     0, // 0 means use all features (preserves accuracy)
		MaxThresholds:   0, // 0 means test all thresholds (preserves accuracy)
	}
}

// Fit the model
func (dt *DecisionTreeClassifier) Fit(X [][]float64, y []float64) {
	if len(X) == 0 || len(y) == 0 || len(X[0]) == 0 {
		// No data to train
		return
	}
	// Store references to original data
	dt.X = X
	dt.Y = y

	// Initialize indices array (all samples)
	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}

	dt.Root = dt.fitRecursive(indices, 0)
}

// Predict returns predictions for each sample
func (dt *DecisionTreeClassifier) Predict(X [][]float64) []float64 {
	preds := make([]float64, len(X))
	for i, x := range X {
		preds[i] = dt.PredictOne(dt.Root, x)
	}
	return preds
}

// Recursive training using indices
func (dt *DecisionTreeClassifier) fitRecursive(indices []int, depth int) *Node {
	nSamples := len(indices)
	if nSamples == 0 {
		return nil
	}

	// Extract labels for these indices
	ySubset := make([]float64, nSamples)
	for i, idx := range indices {
		ySubset[i] = dt.Y[idx]
	}

	// Stop conditions: max depth, pure node, or not enough samples to split
	if depth >= dt.MaxDepth || giniFast(ySubset) == 0 || nSamples < dt.MinSamplesSplit {
		return &Node{
			Value:  majorityClassFast(ySubset),
			IsLeaf: true,
		}
	}

	feature, threshold, found := dt.bestSplit(indices, nSamples)
	if !found {
		// No good split found, make leaf
		return &Node{
			Value:  majorityClassFast(ySubset),
			IsLeaf: true,
		}
	}

	leftIndices, rightIndices := dt.splitIndices(indices, feature, threshold)

	// Defensive: If split produces empty partitions, make leaf
	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		return &Node{
			Value:  majorityClassFast(ySubset),
			IsLeaf: true,
		}
	}

	return &Node{
		FeatureIndex: feature,
		Threshold:    threshold,
		Left:         dt.fitRecursive(leftIndices, depth+1),
		Right:        dt.fitRecursive(rightIndices, depth+1),
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

// Optimized Gini for binary classification (only 2 classes: 0 and 1)
func giniFast(y []float64) float64 {
	n := len(y)
	if n == 0 {
		return 0
	}

	// Count classes (binary: 0 and 1)
	count0 := 0
	count1 := 0
	for _, label := range y {
		if label == 0.0 {
			count0++
		} else {
			count1++
		}
	}

	if count0 == 0 || count1 == 0 {
		return 0 // Pure node
	}

	total := float64(n)
	p0 := float64(count0) / total
	p1 := float64(count1) / total

	return 1.0 - p0*p0 - p1*p1
}

// Optimized majority class for binary classification
func majorityClassFast(y []float64) float64 {
	count0 := 0
	count1 := 0
	for _, label := range y {
		if label == 0.0 {
			count0++
		} else {
			count1++
		}
	}
	if count1 > count0 {
		return 1.0
	}
	return 0.0
}

// Split indices based on feature threshold
func (dt *DecisionTreeClassifier) splitIndices(indices []int, featureIndex int, threshold float64) ([]int, []int) {
	// Pre-allocate with estimated capacity
	leftIndices := make([]int, 0, len(indices)/2)
	rightIndices := make([]int, 0, len(indices)/2)

	for _, idx := range indices {
		if dt.X[idx][featureIndex] <= threshold {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	return leftIndices, rightIndices
}

// Incremental Gini calculation - calculates Gini incrementally while iterating
func incrementalGini(leftCount0, leftCount1, rightCount0, rightCount1 int, totalSamples int) float64 {
	if totalSamples == 0 {
		return 0
	}

	leftTotal := leftCount0 + leftCount1
	rightTotal := rightCount0 + rightCount1

	var leftGini, rightGini float64

	if leftTotal > 0 {
		p0Left := float64(leftCount0) / float64(leftTotal)
		p1Left := float64(leftCount1) / float64(leftTotal)
		leftGini = 1.0 - p0Left*p0Left - p1Left*p1Left
	}

	if rightTotal > 0 {
		p0Right := float64(rightCount0) / float64(rightTotal)
		p1Right := float64(rightCount1) / float64(rightTotal)
		rightGini = 1.0 - p0Right*p0Right - p1Right*p1Right
	}

	p := float64(leftTotal) / float64(totalSamples)
	return p*leftGini + (1-p)*rightGini
}

// bestSplit returns the best feature and threshold for a split using indices
func (dt *DecisionTreeClassifier) bestSplit(indices []int, nSamples int) (int, float64, bool) {
	if nSamples == 0 || len(dt.X) == 0 || len(dt.X[0]) == 0 {
		return 0, 0, false
	}

	nFeatures := len(dt.X[0])

	// Determine how many features to consider
	maxFeaturesToConsider := nFeatures
	if dt.MaxFeatures > 0 {
		maxFeaturesToConsider = dt.MaxFeatures
		if maxFeaturesToConsider > nFeatures {
			maxFeaturesToConsider = nFeatures
		}
	} else if dt.MaxFeatures < 0 {
		// Use sqrt(nFeatures) like Random Forest
		maxFeaturesToConsider = int(math.Sqrt(float64(nFeatures)))
		if maxFeaturesToConsider < 1 {
			maxFeaturesToConsider = 1
		}
	}

	// Create feature indices to consider (sample if needed)
	featureIndices := make([]int, nFeatures)
	for i := range featureIndices {
		featureIndices[i] = i
	}

	// Randomly shuffle and take first maxFeaturesToConsider
	if maxFeaturesToConsider < nFeatures {
		// Fisher-Yates shuffle
		for i := nFeatures - 1; i > 0; i-- {
			j := rand.Intn(i + 1)
			featureIndices[i], featureIndices[j] = featureIndices[j], featureIndices[i]
		}
		featureIndices = featureIndices[:maxFeaturesToConsider]
	}

	bestGini := math.Inf(1)
	var bestFeature int
	var bestThreshold float64
	foundSplit := false

	// Calculate total class counts for this subset
	totalCount0 := 0
	totalCount1 := 0
	for _, idx := range indices {
		if dt.Y[idx] == 0.0 {
			totalCount0++
		} else {
			totalCount1++
		}
	}

	// Check each selected feature
	for _, feature := range featureIndices {
		// Sort indices by feature value for incremental calculation
		sortedIndices := make([]int, nSamples)
		copy(sortedIndices, indices)
		sort.Slice(sortedIndices, func(i, j int) bool {
			return dt.X[sortedIndices[i]][feature] < dt.X[sortedIndices[j]][feature]
		})

		// Extract unique values from sorted indices (more efficient)
		uniqueVals := make([]float64, 0, nSamples)
		lastVal := math.NaN()
		for _, idx := range sortedIndices {
			val := dt.X[idx][feature]
			if math.IsNaN(lastVal) || val != lastVal {
				uniqueVals = append(uniqueVals, val)
				lastVal = val
			}
		}

		// Need at least 2 unique values to split
		if len(uniqueVals) < 2 {
			continue
		}

		// Determine thresholds to test
		thresholds := dt.getThresholds(uniqueVals)

		// Test each threshold using incremental calculation
		leftCount0 := 0
		leftCount1 := 0
		sortedIdx := 0

		// Iterate through thresholds incrementally
		for _, threshold := range thresholds {
			// Advance sortedIdx until we've processed all samples <= threshold
			for sortedIdx < nSamples && dt.X[sortedIndices[sortedIdx]][feature] <= threshold {
				if dt.Y[sortedIndices[sortedIdx]] == 0.0 {
					leftCount0++
				} else {
					leftCount1++
				}
				sortedIdx++
			}

			leftTotal := leftCount0 + leftCount1
			rightCount0 := totalCount0 - leftCount0
			rightCount1 := totalCount1 - leftCount1
			rightTotal := rightCount0 + rightCount1

			// Check min samples split constraint
			if leftTotal < dt.MinSamplesSplit || rightTotal < dt.MinSamplesSplit {
				continue
			}

			// Calculate weighted Gini using incremental method
			g := incrementalGini(leftCount0, leftCount1, rightCount0, rightCount1, nSamples)

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

// getThresholds returns a sampled set of thresholds to test
func (dt *DecisionTreeClassifier) getThresholds(uniqueVals []float64) []float64 {
	nUnique := len(uniqueVals)
	if nUnique < 2 {
		return nil
	}

	// If MaxThresholds is 0 or we have fewer unique values than limit, use all
	if dt.MaxThresholds <= 0 || nUnique-1 <= dt.MaxThresholds {
		thresholds := make([]float64, nUnique-1)
		for i := 0; i < nUnique-1; i++ {
			thresholds[i] = (uniqueVals[i] + uniqueVals[i+1]) / 2.0
		}
		return thresholds
	}

	// Sample thresholds uniformly
	step := float64(nUnique-1) / float64(dt.MaxThresholds)
	thresholds := make([]float64, 0, dt.MaxThresholds)

	for i := 0; i < dt.MaxThresholds; i++ {
		idx := int(float64(i) * step)
		if idx >= nUnique-1 {
			idx = nUnique - 2
		}
		thresholds = append(thresholds, (uniqueVals[idx]+uniqueVals[idx+1])/2.0)
	}

	return thresholds
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
