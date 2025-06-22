package app

import (
	"phishingsms/models/logistic"
	"phishingsms/models/naivebayes"
	"phishingsms/models/tree"
	"sync"
)

type PhishingDetector struct {
	logisticRegression     *logistic.LogisticRegression
	multiNominalNaiveBayes *naivebayes.MultinomialNaiveBayes
	dtree                  *tree.DecisionTreeClassifier
}

type Prediction struct {
	Accuracy  float64
	Precision float64
	Recall    float64
}

func NewPhishingDetector() *PhishingDetector {
	return &PhishingDetector{}
}

func (d *PhishingDetector) Fit(features [][]float64, labels []float64) {
	wg := sync.WaitGroup{}

	linearModel := logistic.NewLogisticRegression(500, 0.01)
	wg.Add(1)
	go func() {
		linearModel.Fit(features, labels)
		wg.Done()
	}()

	d.logisticRegression = linearModel

	mn := naivebayes.NewMultinomialNaiveBayes(labels)
	wg.Add(1)
	go func() {
		mn.Fit(features, labels)
		wg.Done()
	}()

	d.multiNominalNaiveBayes = mn

	dtree := tree.NewDecisionTreeClassifier(25)
	wg.Add(1)
	go func() {
		dtree.Fit(features, labels)
		wg.Done()
	}()

	d.dtree = dtree

	wg.Wait()
}

func (d *PhishingDetector) Predict(features [][]float64) []float64 {
	predict1 := d.logisticRegression.Predict(features)
	predict2 := d.multiNominalNaiveBayes.Predict(features)
	predict3 := d.dtree.Predict(features)

	predictions := make([]float64, len(features))

	for i := range features {
		predictedValue := d.ensemble([]float64{predict1[i], predict2[i], predict3[i]})

		predictions[i] = predictedValue
	}

	return predictions
}

func (d *PhishingDetector) Stats(predictions []float64, labels []float64) *Prediction {
	prediction := Prediction{}

	tp, fp, fn := 0.0, 0.0, 0.0

	for i, predictedValue := range predictions {

		if predictedValue == labels[i] {
			prediction.Accuracy += 1.0
		}

		if predictedValue == 1 && labels[i] == 1 {
			tp += 1.0 // True Positive
		} else if predictedValue == 1 && labels[i] == 0 {
			fp += 1.0 // False Positive
		} else if predictedValue == 0 && labels[i] == 1 {
			fn += 1.0 // False Negative
		}
	}

	if tp+fp > 0 {
		prediction.Precision = tp / (tp + fp) // Precision = TP / (TP + FP)
	}

	if tp+fn > 0 {
		prediction.Recall = tp / (tp + fn) // Recall = TP / (TP + FN)
	}

	prediction.Accuracy /= float64(len(labels))

	return &prediction
}

func (d *PhishingDetector) ensemble(predictions []float64) float64 {
	if len(predictions) != 3 {
		panic("Expected 3 predictions")
	}

	sum := 0.0

	for _, prediction := range predictions {
		sum += prediction
	}

	// hard voting
	if sum >= 2 {
		return 1
	}

	return 0
}
