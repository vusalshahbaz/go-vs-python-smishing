package naivebayes

type MultinomialNaiveBayes struct {
	classes       []float64
	featureCounts map[float64]map[float64]int
	totalCounts   map[float64]int
}

func NewMultinomialNaiveBayes(classes []float64) *MultinomialNaiveBayes {
	return &MultinomialNaiveBayes{
		classes:       classes,
		featureCounts: make(map[float64]map[float64]int),
		totalCounts:   make(map[float64]int),
	}
}

func (mnb *MultinomialNaiveBayes) Fit(features [][]float64, labels []float64) {
	for i, label := range labels {
		if _, exists := mnb.featureCounts[label]; !exists {
			mnb.featureCounts[label] = make(map[float64]int)
		}

		for _, feature := range features[i] {
			mnb.featureCounts[label][feature]++
			mnb.totalCounts[label]++
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
	maxProb := -1.0
	var bestClass float64

	for _, class := range mnb.classes {
		prob := mnb.classProbability(class, features)
		if prob > maxProb {
			maxProb = prob
			bestClass = class
		}
	}

	return bestClass
}

func (mnb *MultinomialNaiveBayes) classProbability(class float64, features []float64) float64 {
	prob := 1.0
	totalCount := mnb.totalCounts[class]

	for _, feature := range features {
		count := mnb.featureCounts[class][feature]
		prob *= float64(count+1) / float64(totalCount+len(mnb.featureCounts[class]))
	}

	return prob
}
