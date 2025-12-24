package naivebayes

import (
	"math"
)

// MultinomialNaiveBayes works with TF-IDF features by treating them as counts
// This matches sklearn's MultinomialNB behavior
type MultinomialNaiveBayes struct {
	classes       []float64
	featureCounts map[float64][]float64 // class -> feature counts (sum of TF-IDF values per feature index)
	totalCounts   map[float64]float64   // class -> total count (sum of all TF-IDF values)
	classProbs    map[float64]float64   // prior probability of each class
	numFeatures   int
	alpha         float64 // smoothing parameter (Laplace smoothing), default 1.0 like sklearn
}

func NewMultinomialNaiveBayes(classes []float64) *MultinomialNaiveBayes {
	return &MultinomialNaiveBayes{
		classes:       classes,
		featureCounts: make(map[float64][]float64),
		totalCounts:   make(map[float64]float64),
		classProbs:    make(map[float64]float64),
		alpha:         1.0, // default smoothing parameter (matches sklearn)
	}
}

func (mnb *MultinomialNaiveBayes) Fit(features [][]float64, labels []float64) {
	if len(features) == 0 {
		return
	}

	mnb.numFeatures = len(features[0])

	// Count samples per class
	classCounts := make(map[float64]int)
	for _, label := range labels {
		classCounts[label]++
	}

	totalSamples := float64(len(labels))

	// Calculate prior probabilities
	for _, class := range mnb.classes {
		mnb.classProbs[class] = float64(classCounts[class]) / totalSamples
	}

	// Initialize feature counts for each class (per feature index, not map keys!)
	for _, class := range mnb.classes {
		mnb.featureCounts[class] = make([]float64, mnb.numFeatures)
		mnb.totalCounts[class] = 0.0
	}

	// Accumulate TF-IDF values as "counts" for each class and feature INDEX
	// This is how MultinomialNB works with TF-IDF - it treats the values as counts
	for i, label := range labels {
		for j := 0; j < mnb.numFeatures; j++ {
			// Treat TF-IDF value as a count (they're non-negative)
			value := features[i][j]
			if value > 0 {
				mnb.featureCounts[label][j] += value
				mnb.totalCounts[label] += value
			}
		}
	}
}

func (mnb *MultinomialNaiveBayes) Predict(features [][]float64) []float64 {
	predictions := make([]float64, len(features))

	for i, featureSet := range features {
		predictions[i] = mnb.PredictSingle(featureSet)
	}

	return predictions
}

func (mnb *MultinomialNaiveBayes) PredictSingle(features []float64) float64 {
	maxLogProb := math.Inf(-1)
	var bestClass float64

	for _, class := range mnb.classes {
		logProb := mnb.classLogProbability(class, features)
		if logProb > maxLogProb {
			maxLogProb = logProb
			bestClass = class
		}
	}

	return bestClass
}

func (mnb *MultinomialNaiveBayes) classLogProbability(class float64, features []float64) float64 {
	// Use log probabilities to avoid underflow
	// P(class | features) ∝ P(class) * ∏ P(feature_i | class)^value_i

	// Start with log of prior probability
	logProb := math.Log(mnb.classProbs[class] + 1e-10) // small epsilon to avoid log(0)

	totalCount := mnb.totalCounts[class]
	alpha := mnb.alpha
	numFeatures := float64(mnb.numFeatures)

	// For each feature, calculate log P(feature_i | class)
	// P(feature_i | class) = (count_i + alpha) / (total_count + alpha * num_features)
	// In log space: log(count_i + alpha) - log(total_count + alpha * num_features)
	// But we multiply by the feature value (treated as count), so:
	// log(P(feature_i | class)^value_i) = value_i * log(P(feature_i | class))

	denominator := totalCount + alpha*numFeatures
	logDenominator := math.Log(denominator)

	for j := 0; j < mnb.numFeatures && j < len(features); j++ {
		value := features[j]
		if value > 0 {
			// Get the count for this feature index in this class
			featureCount := mnb.featureCounts[class][j]

			// Calculate P(feature_j | class) with smoothing
			// P(feature_j | class) = (featureCount + alpha) / (totalCount + alpha * numFeatures)
			numerator := featureCount + alpha
			logNumerator := math.Log(numerator)

			// log(P(feature_j | class)^value) = value * log(P(feature_j | class))
			// = value * (log(numerator) - log(denominator))
			logProb += value * (logNumerator - logDenominator)
		}
	}

	return logProb
}
