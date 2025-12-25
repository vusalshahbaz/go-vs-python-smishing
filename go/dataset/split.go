package dataset

import (
	"math/rand/v2"
)

// TrainTestSplit performs a stratified split to ensure both classes are present in train and test sets
// This maintains approximately the same class distribution in both sets
func TrainTestSplit(features [][]float64, labels []float64, trainRatio float64) ([][]float64, []float64, [][]float64, []float64) {
	// Separate indices by class to ensure stratified split
	class0Indices := []int{}
	class1Indices := []int{}

	for i, label := range labels {
		if label == 0.0 {
			class0Indices = append(class0Indices, i)
		} else {
			class1Indices = append(class1Indices, i)
		}
	}

	// Shuffle each class's indices independently (using seed 42 to match Python's random_state=42)
	r := rand.New(rand.NewPCG(42, 0))
	r.Shuffle(len(class0Indices), func(i, j int) {
		class0Indices[i], class0Indices[j] = class0Indices[j], class0Indices[i]
	})
	r.Shuffle(len(class1Indices), func(i, j int) {
		class1Indices[i], class1Indices[j] = class1Indices[j], class1Indices[i]
	})

	// Calculate split sizes for each class (maintains class distribution)
	// Account for rounding loss: calculate total expected train size first,
	// then adjust one of the sizes to match the expected total
	totalSamples := len(class0Indices) + len(class1Indices)
	expectedTrainSize := int(float64(totalSamples) * trainRatio)

	trainSize0 := int(float64(len(class0Indices)) * trainRatio)
	trainSize1 := int(float64(len(class1Indices)) * trainRatio)

	// Adjust for rounding loss to match expected total train size
	actualTrainSize := trainSize0 + trainSize1
	if actualTrainSize < expectedTrainSize {
		// Add the difference to trainSize1 to compensate for rounding loss
		trainSize1 += expectedTrainSize - actualTrainSize
	}

	// Collect train and test indices
	trainIndices := make([]int, 0, trainSize0+trainSize1)
	testIndices := make([]int, 0, len(class0Indices)+len(class1Indices)-trainSize0-trainSize1)

	// Add class 0 samples (maintains proportion)
	trainIndices = append(trainIndices, class0Indices[:trainSize0]...)
	testIndices = append(testIndices, class0Indices[trainSize0:]...)

	// Add class 1 samples (maintains proportion)
	trainIndices = append(trainIndices, class1Indices[:trainSize1]...)
	testIndices = append(testIndices, class1Indices[trainSize1:]...)

	// Shuffle train and test indices to mix classes
	r.Shuffle(len(trainIndices), func(i, j int) {
		trainIndices[i], trainIndices[j] = trainIndices[j], trainIndices[i]
	})
	r.Shuffle(len(testIndices), func(i, j int) {
		testIndices[i], testIndices[j] = testIndices[j], testIndices[i]
	})

	// Build train sets with deep copy
	trainFeatures := make([][]float64, len(trainIndices))
	trainLabels := make([]float64, len(trainIndices))
	for i, idx := range trainIndices {
		trainFeatures[i] = make([]float64, len(features[idx]))
		copy(trainFeatures[i], features[idx])
		trainLabels[i] = labels[idx]
	}

	// Build test sets with deep copy
	testFeatures := make([][]float64, len(testIndices))
	testLabels := make([]float64, len(testIndices))
	for i, idx := range testIndices {
		testFeatures[i] = make([]float64, len(features[idx]))
		copy(testFeatures[i], features[idx])
		testLabels[i] = labels[idx]
	}

	return trainFeatures, trainLabels, testFeatures, testLabels
}
