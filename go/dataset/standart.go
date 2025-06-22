package dataset

import "math"

func StandardizeFeatures(features [][]float64) ([][]float64, []float64, []float64) {
	numSamples := len(features)
	numFeatures := len(features[0])
	means := make([]float64, numFeatures)
	stds := make([]float64, numFeatures)

	// Compute mean
	for j := 0; j < numFeatures; j++ {
		sum := 0.0
		for i := 0; i < numSamples; i++ {
			sum += features[i][j]
		}
		means[j] = sum / float64(numSamples)
	}

	// Compute std deviation
	for j := 0; j < numFeatures; j++ {
		var sumSquares float64
		for i := 0; i < numSamples; i++ {
			diff := features[i][j] - means[j]
			sumSquares += diff * diff
		}
		stds[j] = math.Sqrt(sumSquares / float64(numSamples))
	}

	// Apply standardization
	standardized := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		standardized[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			if stds[j] != 0 {
				standardized[i][j] = (features[i][j] - means[j]) / stds[j]
			} else {
				standardized[i][j] = 0
			}
		}
	}

	return standardized, means, stds
}
