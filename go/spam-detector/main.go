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
	"spam-detector/detector"
	"time"
)

func main() {
	datasetPath := "../../datasets/spam-sms.csv"
	if len(os.Args) > 1 {
		datasetPath = os.Args[1]
	}

	features, labels := LoadPhishingData(datasetPath, map[string]float64{"ham": 0, "spam": 1})

	vectorizer := tfidf.New(2000, features)

	vectorizedFeatures := vectorizer.Transform(features)

	vectorizedFeatures, _, _ = dataset.StandardizeFeatures(vectorizedFeatures)

	XTrain, YTrain, XTest, YTest := dataset.TrainTestSplit(vectorizedFeatures, labels, 0.8)

	spamDetector := detector.NewSpamDetector()

	start := time.Now()
	spamDetector.Fit(XTrain, YTrain)
	elapsed := time.Since(start)

	fmt.Println("Time taken to fit the model:", elapsed)

	start = time.Now()
	predictions := spamDetector.Predict(XTest)

	elapsed = time.Since(start)
	fmt.Println("Time taken to predict:", elapsed)

	prediction := spamDetector.Stats(predictions, YTest)

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

	runHttpServer(vectorizer, spamDetector)
}

func runHttpServer(vectorizer *tfidf.Vectorizer, spamDetector *detector.SpamDetector) {
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
			"predictions": spamDetector.Predict(vectorized),
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

func LoadPhishingData(filename string, classess map[string]float64) ([]string, []float64) {
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

	features := make([]string, len(records)) // Skip header
	labels := make([]float64, len(records))

	for i, record := range records { // Skip header

		features[i] = record[1]

		label, ok := classess[record[0]]
		if !ok {
			panic(fmt.Sprintf("Unknown class: %s", record[0]))
		}

		labels[i] = label
	}

	return features, labels
}
