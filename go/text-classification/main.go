package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"github.com/vusalshahbaz/go-ml/dataset"
	"github.com/vusalshahbaz/go-ml/vectorizers/tfidf"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
	"text-classification/classifier"
	"time"
)

func main() {
	features, labels := LoadTextData("../datasets/IMDB Dataset.csv", map[string]float64{"negative": 0, "positive": 1})

	vectorizer := tfidf.New(2000, features)

	vectorizedFeatures := vectorizer.Transform(features)

	vectorizedFeatures, _, _ = dataset.StandardizeFeatures(vectorizedFeatures)

	XTrain, YTrain, XTest, YTest := dataset.TrainTestSplit(vectorizedFeatures, labels, 0.8)

	textClassifier := detector.NewTextClassifier()

	start := time.Now()
	textClassifier.Fit(XTrain, YTrain)
	elapsed := time.Since(start)

	fmt.Println("Time taken to fit the model:", elapsed)

	start = time.Now()
	fmt.Println("Predicting...")
	predictions := textClassifier.Predict(XTest)

	elapsed = time.Since(start)
	fmt.Println("Time taken to predict:", elapsed)

	prediction := textClassifier.Stats(predictions, YTest)

	fmt.Println("Accuracy:", prediction.Accuracy)
	fmt.Println("Precision:", prediction.Precision)
	fmt.Println("Recall:", prediction.Recall)
	fmt.Println("Macro Precision:", prediction.MacroPrecision)
	fmt.Println("Macro Recall:", prediction.MacroRecall)
	fmt.Println("Macro F1:", prediction.MacroF1)
	fmt.Println("Confusion Matrix:")
	fmt.Printf("  Actual\\Predicted   0      1\n")
	fmt.Printf("  0                  %.0f     %.0f\n", prediction.ConfusionMatrix["0"]["0"], prediction.ConfusionMatrix["0"]["1"])
	fmt.Printf("  1                  %.0f     %.0f\n", prediction.ConfusionMatrix["1"]["0"], prediction.ConfusionMatrix["1"]["1"])

	runHttpServer(vectorizer, textClassifier)
}

func runHttpServer(vectorizer *tfidf.Vectorizer, textClassifier *detector.TextClassifier) {
	type Request struct {
		Message string `json:"message"`
	}

	http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		var req Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		text := req.Message

		vectorized := vectorizer.Transform([]string{text})

		res := map[string]interface{}{
			"predictions": textClassifier.Predict(vectorized),
		}

		fmt.Println(text, res)

		w.Header().Set("Content-Type", "application/json")

		if err := json.NewEncoder(w).Encode(res); err != nil {
			http.Error(w, "Error encoding response", http.StatusInternalServerError)
			return
		}
	})

	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		panic(err)
	}
}

type randomSource struct{}

func (r *randomSource) Uint64() uint64 {
	return rand.Uint64()
}

func LoadTextData(filename string, classes map[string]float64) ([]string, []float64) {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	defer file.Close()

	records, err := csv.NewReader(file).ReadAll()
	if err != nil {
		panic(err)
	}

	records = records[1:] // Skip header

	r := rand.New(&randomSource{})

	r.Shuffle(len(records), func(i, j int) {
		records[i], records[j] = records[j], records[i]
	})

	features := make([]string, len(records))
	labels := make([]float64, len(records))

	for i, record := range records {
		// For IMDB format: review is in column 0, sentiment is in column 1
		features[i] = strings.ReplaceAll(strings.ReplaceAll(record[0], "<br /><br />", " "), "-", " ")

		label, ok := classes[record[1]]
		if !ok {
			panic(fmt.Sprintf("Unknown class: %s", record[1]))
		}

		labels[i] = label
	}

	return features, labels
}
