package logistic

import (
	"fmt"
	"math"
)

type LogisticRegression struct {
	iterations   int
	learningRate float64
	weights      []float64
	bias         float64
}

func NewLogisticRegression(iterations int, learningRate float64) *LogisticRegression {
	return &LogisticRegression{
		iterations:   iterations,
		learningRate: learningRate,
	}
}

func (logistic *LogisticRegression) Fit(features [][]float64, labels []float64) {
	numSamples := len(features)
	numFeatures := len(features[0])

	logistic.weights = make([]float64, numFeatures)
	logistic.bias = 0.0

	for i := 0; i < logistic.iterations; i++ {
		logistic.gradientDescent(features, labels, numSamples, numFeatures)
	}
}

func (logistic *LogisticRegression) gradientDescent(features [][]float64, labels []float64, numSamples int, numFeatures int) {
	if numSamples == 0 {
		fmt.Println("Error: numSamples is zero, cannot perform gradient descent.")
		return
	}

	weightGrad := make([]float64, numFeatures)
	biasGrad := 0.0

	for i := 0; i < numSamples; i++ {
		prediction := logistic.PredictSingle(features[i])
		e := prediction - labels[i]

		for j := 0; j < numFeatures; j++ {
			weightGrad[j] += e * features[i][j]
		}
		biasGrad += e
	}

	for j := 0; j < numFeatures; j++ {
		logistic.weights[j] -= logistic.learningRate * weightGrad[j] / float64(numSamples)
	}
	logistic.bias -= logistic.learningRate * biasGrad / float64(numSamples)
}

func (logistic *LogisticRegression) GetWeights() []float64 {
	return logistic.weights
}

func (logistic *LogisticRegression) GetBias() float64 {
	return logistic.bias
}

func (logistic *LogisticRegression) PredictSingle(features []float64) float64 {
	prediction := logistic.bias
	for j := 0; j < len(features); j++ {
		prediction += logistic.weights[j] * features[j]
	}

	return sigmoid(prediction)

}

func (logistic *LogisticRegression) PredictSingleBinary(features []float64) float64 {
	prediction := logistic.PredictSingle(features)
	if prediction >= 0.5 {
		return 1.0 // Positive class
	}
	return 0.0 // Negative class
}

func (logistic *LogisticRegression) Predict(features [][]float64) []float64 {
	predictions := make([]float64, len(features))
	for i, feature := range features {
		predictions[i] = logistic.PredictSingle(feature)
	}
	return predictions
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
