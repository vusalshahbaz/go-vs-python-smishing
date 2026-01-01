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
	"phishingsms/internal/phishing-email-detector/detector"
	"strings"
	"time"
)

func main() {
	features, labels := LoadPhishingEmailData("./ling.csv", map[string]float64{"0": 0, "1": 1})

	vectorizer := tfidf.New(2000, features)

	vectorizedFeatures := vectorizer.Transform(features)

	vectorizedFeatures, _, _ = dataset.StandardizeFeatures(vectorizedFeatures)

	XTrain, YTrain, XTest, YTest := dataset.TrainTestSplit(vectorizedFeatures, labels, 0.8)

	phishingDetector := detector.NewPhishingEmailDetector()

	start := time.Now()
	phishingDetector.Fit(XTrain, YTrain)
	elapsed := time.Since(start)

	fmt.Println("Time taken to fit the model:", elapsed)

	start = time.Now()
	predictions := phishingDetector.Predict(XTest)

	elapsed = time.Since(start)
	fmt.Println("Time taken to predict:", elapsed)

	prediction := phishingDetector.Stats(predictions, YTest)

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

	runHttpServer(vectorizer, phishingDetector)
}

func runHttpServer(vectorizer *tfidf.Vectorizer, phishingDetector *detector.PhishingEmailDetector) {
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
			"predictions": phishingDetector.Predict(vectorized),
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

func LoadPhishingEmailData(filename string, classes map[string]float64) ([]string, []float64) {
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

	// Filter out malformed rows first
	validRecords := make([][]string, 0, len(records))
	for _, record := range records {
		if len(record) >= 3 {
			validRecords = append(validRecords, record)
		}
	}

	r := rand.New(&randomSource{})

	r.Shuffle(len(validRecords), func(i, j int) {
		validRecords[i], validRecords[j] = validRecords[j], validRecords[i]
	})

	features := make([]string, len(validRecords))
	labels := make([]float64, len(validRecords))

	for i, record := range validRecords {
		// CSV format: subject (col 1, index 0), body (col 2, index 1), label (col 3, index 2)
		// We need columns 2 and 3 (1-indexed) = indices 1 and 2 (0-indexed)
		// Ignore subject (column 1/index 0), use body (column 2/index 1) and label (column 3/index 2)
		if len(record) < 3 {
			continue // Skip if not enough columns
		}

		// Column 2 (1-indexed) = index 1 (0-indexed) = body
		features[i] = record[1]

		// Column 3 (1-indexed) = index 2 (0-indexed) = label
		labelStr := strings.TrimSpace(record[2])
		label, ok := classes[labelStr]
		if !ok {
			panic(fmt.Sprintf("Unknown class: %s", labelStr))
		}

		labels[i] = label
	}

	return features, labels
}
