package detector

import (
	"fmt"
	"github.com/vusalshahbaz/go-ml/models/logistic"
	"github.com/vusalshahbaz/go-ml/models/naivebayes"
	"github.com/vusalshahbaz/go-ml/models/tree"
	"sync"
)

type SpamDetector struct {
	logisticRegression     *logistic.LogisticRegression
	multiNominalNaiveBayes *naivebayes.MultinomialNaiveBayes
	dtree                  *tree.DecisionTreeClassifier
}

type Prediction struct {
	Accuracy        float64
	Precision       float64
	Recall          float64
	MacroRecall     float64
	MacroPrecision  float64
	MacroF1         float64
	ConfusionMatrix map[string]map[string]float64
}

func NewSpamDetector() *SpamDetector {
	return &SpamDetector{}
}

func (d *SpamDetector) Fit(features [][]float64, labels []float64) {
	wg := sync.WaitGroup{}
	total := len(features)

	fmt.Printf("spamDetector.Fit() - total: %d, starting training...\n", total)

	linearModel := logistic.NewLogisticRegression(500, 0.01)
	wg.Add(1)
	go func() {
		fmt.Printf("spamDetector.Fit() - training LogisticRegression model...\n")
		linearModel.Fit(features, labels)
		fmt.Printf("spamDetector.Fit() - LogisticRegression model completed\n")
		wg.Done()
	}()

	d.logisticRegression = linearModel

	mn := naivebayes.NewMultinomialNaiveBayes(labels)
	wg.Add(1)
	go func() {
		fmt.Printf("spamDetector.Fit() - training MultinomialNaiveBayes model...\n")
		mn.Fit(features, labels)
		fmt.Printf("spamDetector.Fit() - MultinomialNaiveBayes model completed\n")
		wg.Done()
	}()

	d.multiNominalNaiveBayes = mn

	dtree := tree.NewDecisionTreeClassifier(25)
	wg.Add(1)
	go func() {
		fmt.Printf("spamDetector.Fit() - training DecisionTreeClassifier model...\n")
		dtree.Fit(features, labels)
		fmt.Printf("spamDetector.Fit() - DecisionTreeClassifier model completed\n")
		wg.Done()
	}()

	d.dtree = dtree

	wg.Wait()
	fmt.Printf("spamDetector.Fit() - all models trained, processed: %d\n", total)
}

func (d *SpamDetector) Predict(features [][]float64) []float64 {
	wg := sync.WaitGroup{}
	total := len(features)

	wg.Add(3)

	var predict1 []float64
	var predict2 []float64
	var predict3 []float64

	go func() {
		predict1 = d.logisticRegression.Predict(features)
		wg.Done()
	}()

	go func() {
		predict2 = d.multiNominalNaiveBayes.Predict(features)
		wg.Done()
	}()

	go func() {
		predict3 = d.dtree.Predict(features)
		wg.Done()
	}()

	wg.Wait()

	predictions := make([]float64, len(features))

	for i := range features {
		predictedValue := d.ensemble([]float64{predict1[i], predict2[i], predict3[i]})

		predictions[i] = predictedValue

		// Progress logging
		if (i+1)%100 == 0 || i+1 == total {
			fmt.Printf("\rspamDetector.Predict() - total: %d, processed: %d", total, i+1)
		}
	}

	fmt.Println() // New line after progress
	return predictions
}

func (d *SpamDetector) Stats(predictions []float64, labels []float64) *Prediction {
	prediction := Prediction{}

	// Confusion matrix: [actual][predicted]
	confusionMatrix := make(map[string]map[string]float64)
	confusionMatrix["0"] = make(map[string]float64) // actual class 0
	confusionMatrix["1"] = make(map[string]float64) // actual class 1

	tp, fp, fn, tn := 0.0, 0.0, 0.0, 0.0

	// Class-specific metrics
	tp0, fp0, fn0 := 0.0, 0.0, 0.0 // For class 0
	tp1, fp1, fn1 := 0.0, 0.0, 0.0 // For class 1

	for i, predictedValue := range predictions {
		actualValue := labels[i]

		if predictedValue == actualValue {
			prediction.Accuracy += 1.0
		}

		// Update confusion matrix
		predStr := formatFloat(predictedValue)
		actualStr := formatFloat(actualValue)
		confusionMatrix[actualStr][predStr]++

		// Binary classification metrics (class 1 as positive)
		if predictedValue == 1 && actualValue == 1 {
			tp += 1.0 // True Positive
		} else if predictedValue == 1 && actualValue == 0 {
			fp += 1.0 // False Positive
		} else if predictedValue == 0 && actualValue == 1 {
			fn += 1.0 // False Negative
		} else if predictedValue == 0 && actualValue == 0 {
			tn += 1.0 // True Negative
		}

		// Class 0 metrics (treating 0 as positive)
		if predictedValue == 0 && actualValue == 0 {
			tp0 += 1.0
		} else if predictedValue == 0 && actualValue == 1 {
			fp0 += 1.0
		} else if predictedValue == 1 && actualValue == 0 {
			fn0 += 1.0
		}

		// Class 1 metrics (treating 1 as positive)
		if predictedValue == 1 && actualValue == 1 {
			tp1 += 1.0
		} else if predictedValue == 1 && actualValue == 0 {
			fp1 += 1.0
		} else if predictedValue == 0 && actualValue == 1 {
			fn1 += 1.0
		}
	}

	// Binary precision and recall (class 1 as positive)
	if tp+fp > 0 {
		prediction.Precision = tp / (tp + fp)
	}
	if tp+fn > 0 {
		prediction.Recall = tp / (tp + fn)
	}

	// Class-specific precision and recall
	precision0 := 0.0
	if tp0+fp0 > 0 {
		precision0 = tp0 / (tp0 + fp0)
	}
	recall0 := 0.0
	if tp0+fn0 > 0 {
		recall0 = tp0 / (tp0 + fn0)
	}

	precision1 := 0.0
	if tp1+fp1 > 0 {
		precision1 = tp1 / (tp1 + fp1)
	}
	recall1 := 0.0
	if tp1+fn1 > 0 {
		recall1 = tp1 / (tp1 + fn1)
	}

	// Macro-averaged metrics
	prediction.MacroPrecision = (precision0 + precision1) / 2.0
	prediction.MacroRecall = (recall0 + recall1) / 2.0

	// Macro F1
	f10 := 0.0
	if precision0+recall0 > 0 {
		f10 = 2 * (precision0 * recall0) / (precision0 + recall0)
	}
	f11 := 0.0
	if precision1+recall1 > 0 {
		f11 = 2 * (precision1 * recall1) / (precision1 + recall1)
	}
	prediction.MacroF1 = (f10 + f11) / 2.0

	prediction.Accuracy /= float64(len(labels))

	// Confusion matrix is already built in the loop
	prediction.ConfusionMatrix = confusionMatrix

	return &prediction
}

func formatFloat(val float64) string {
	if val == 0.0 {
		return "0"
	}
	return "1"
}

func (d *SpamDetector) ensemble(predictions []float64) float64 {
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
