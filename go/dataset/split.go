package dataset

func TrainTestSplit(features [][]float64, labels []float64, trainRatio float64) ([][]float64, []float64, [][]float64, []float64) {
	trainSize := int(float64(len(features)) * trainRatio)

	trainFeatures := make([][]float64, trainSize)
	trainLabels := make([]float64, trainSize)

	testFeatures := make([][]float64, len(features)-trainSize)
	testLabels := make([]float64, len(labels)-trainSize)

	copy(trainFeatures, features[:trainSize])
	copy(trainLabels, labels[:trainSize])

	copy(testFeatures, features[trainSize:])
	copy(testLabels, labels[trainSize:])

	return trainFeatures, trainLabels, testFeatures, testLabels
}
