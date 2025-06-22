package tfidf

import (
	"regexp"
	"slices"
	"strings"
)

// Global vocabulary (word -> ID)
var vocab = make(map[string]int)
var vocabIndex = 0

type Document struct {
	Text string
}

var stopwords = []string{
	"i",
	"me",
	"my",
	"myself",
	"we",
	"our",
	"ours",
	"ourselves",
	"you",
	"your",
	"yours",
	"yourself",
	"yourselves",
	"he",
	"him",
	"his",
	"himself",
	"she",
	"her",
	"hers",
	"herself",
	"it",
	"its",
	"itself",
	"they",
	"them",
	"their",
	"theirs",
	"themselves",
	"what",
	"which",
	"who",
	"whom",
	"this",
	"that",
	"these",
	"those",
	"am",
	"is",
	"are",
	"was",
	"were",
	"be",
	"been",
	"being",
	"have",
	"has",
	"had",
	"having",
	"do",
	"does",
	"did",
	"doing",
	"a",
	"an",
	"the",
	"and",
	"but",
	"if",
	"or",
	"because",
	"as",
	"until",
	"while",
	"of",
	"at",
	"by",
	"for",
	"with",
	"about",
	"against",
	"between",
	"into",
	"through",
	"during",
	"before",
	"after",
	"above",
	"below",
	"to",
	"from",
	"up",
	"down",
	"in",
	"out",
	"on",
	"off",
	"over",
	"under",
	"again",
	"further",
	"then",
	"once",
	"here",
	"there",
	"when",
	"where",
	"why",
	"how",
	"all",
	"any",
	"both",
	"each",
	"few",
	"more",
	"most",
	"other",
	"some",
	"such",
	"no",
	"nor",
	"not",
	"only",
	"own",
	"same",
	"so",
	"than",
	"too",
	"very",
	"s",
	"t",
	"can",
	"will",
	"just",
	"don",
	"should",
	"now",
}

func (d Document) IDs() []int {
	words := strings.Split(cleanText(d.Text), " ")

	var ids []int
	for _, word := range words {
		if len(word) < 3 {
			continue
		}

		// Skip stopwords
		if slices.Contains(stopwords, word) {
			continue
		}

		// Add to vocabulary if it's new
		if _, exists := vocab[word]; !exists {
			vocab[word] = vocabIndex
			vocabIndex++
		}

		ids = append(ids, vocab[word])
	}
	return ids
}

var re = regexp.MustCompile(`[^a-z\s]+`)

func cleanText(text string) string {
	cleaned := re.ReplaceAllString(strings.ToLower(text), "")
	return strings.TrimSpace(cleaned)
}
