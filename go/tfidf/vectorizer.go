package tfidf

import (
	"github.com/go-nlp/tfidf"
	"sort"
)

type wordScore struct {
	word  int
	score float64
}

type Vectorizer struct {
	model       *tfidf.TFIDF
	maxFeatures int

	wordScores []wordScore
}

func New(maxFeatures int, texts []string) *Vectorizer {
	model, wordScores := getTfidf(maxFeatures, texts)

	return &Vectorizer{model: model, maxFeatures: maxFeatures, wordScores: wordScores}
}

func getTfidf(features int, texts []string) (*tfidf.TFIDF, []wordScore) {
	tf := tfidf.New()

	for _, t := range texts {
		tf.Add(Document{Text: t})
	}

	tf.CalculateIDF()

	var tfidfList []wordScore

	for word, tfVal := range tf.TF {
		if idfVal, ok := tf.IDF[word]; ok {
			tfidfList = append(tfidfList, wordScore{
				word:  word,
				score: tfVal * idfVal,
			})
		}
	}

	// Sort by score descending
	sort.Slice(tfidfList, func(i, j int) bool {
		return tfidfList[i].score > tfidfList[j].score
	})

	// Select top 3000 or fewer
	if len(tfidfList) > features {
		tfidfList = tfidfList[:features]
	}

	return tf, tfidfList
}

func (v *Vectorizer) Transform(texts []string) [][]float64 {
	var result [][]float64

	for _, text := range texts {
		scores := v.Scan(text)
		result = append(result, scores)
	}

	return result
}

func (v *Vectorizer) Scan(text string) []float64 {
	document := Document{Text: text}

	ids := document.IDs()
	tfidfs := v.model.Score(document)

	scores := make([]float64, len(v.wordScores))

	for i, ws := range v.wordScores {
		scores[i] = 0.0

		for j, id := range ids {
			if ws.word == id {
				scores[i] = tfidfs[j]
				break
			}
		}
	}

	return scores
}
