package dataset

import (
	"encoding/csv"
	"math/rand/v2"
	"os"
)

type randomSource struct{}

func (r *randomSource) Uint64() uint64 {
	return rand.Uint64()
}

func LoadPhishingData(filename string) ([]string, []float64) {
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

		val := record[0]
		if val == "ham" {
			labels[i] = 0 // Malignant
		}

		if val == "spam" {
			labels[i] = 1 // Benign
		}
	}

	return features, labels
}
