package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"phishingsms/app"
	"phishingsms/dataset"
	"phishingsms/tfidf"
	"time"
)

func main() {
	features, labels := dataset.LoadPhishingData("./spam.csv")

	vectorizer := tfidf.New(2000, features)

	vectorizedFeatures := vectorizer.Transform(features)

	vectorizedFeatures, _, _ = dataset.StandardizeFeatures(vectorizedFeatures)

	XTrain, YTrain, XTest, YTest := dataset.TrainTestSplit(vectorizedFeatures, labels, 0.8)

	detector := app.NewPhishingDetector()

	start := time.Now()
	detector.Fit(XTrain, YTrain)
	elapsed := time.Since(start)

	fmt.Println("Time taken to fit the model:", elapsed)

	start = time.Now()
	predictions := detector.Predict(XTest)

	elapsed = time.Since(start)
	fmt.Println("Time taken to predict:", elapsed)

	prediction := detector.Stats(predictions, YTest)

	fmt.Println("Accuracy:", prediction.Accuracy)
	fmt.Println("Precision:", prediction.Precision)
	fmt.Println("Recall:", prediction.Recall)

	runHttpServer(vectorizer, detector)
}

func runHttpServer(vectorizer *tfidf.Vectorizer, detector *app.PhishingDetector) {
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
			"predictions": detector.Predict(vectorized),
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
