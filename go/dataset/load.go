package dataset

import (
	"encoding/csv"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
)

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
