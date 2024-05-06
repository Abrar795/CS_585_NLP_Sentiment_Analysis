from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import sys

class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_total_counts = defaultdict(int)
        self.vocabulary = set()

    def fit(self, X_train, y_train, alpha=1):
        # Count occurrences of words in each class
        for document, label in zip(X_train, y_train):
            for word in document:
                self.class_word_counts[label][word] += 1
                self.class_total_counts[label] += 1
                self.vocabulary.add(word)

        # Calculate probabilities with add-1 smoothing
        self.params = {}
        for label in self.class_word_counts:
            self.params[label] = {}
            total_word_count = sum(self.class_word_counts[label].values())
            for word in self.vocabulary:
                word_count = self.class_word_counts[label][word]
                self.params[label][word] = (word_count + alpha) / (total_word_count + alpha * len(self.vocabulary))

    def predict(self, X_test):
        predictions = []
        probabilities = []  # Store probabilities for each class
        for document in X_test:
            scores = {label: np.log(self.prior(label)) for label in self.class_word_counts}
            for word in document:
                for label in self.class_word_counts:
                    if word in self.params[label]:
                        scores[label] += np.log(self.params[label][word])
            # Calculate probabilities from scores
            prob_scores = {label: np.exp(score) for label, score in scores.items()}
            probabilities.append(prob_scores)
            predicted_label = max(scores, key=scores.get)
            predictions.append(predicted_label)
        return predictions, probabilities

    def prior(self, label):
        return self.class_total_counts[label] / sum(self.class_total_counts.values())

if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 2:
        print("Default training size is 80 %")
        input_number = 80;

    
    # Retrieve the input number from the command-line argument and convert it to float
    if len(sys.argv) > 1:
        try:
            input_number = float(sys.argv[1])
            if not (20 <= input_number <= 80):
                input_number = 80
        except ValueError:
            input_number = 80
            

    # Print the input number
    print("Abrar Ahmed, Mohammed, A20540340 solution:")
    print("Training size:", input_number,"%")
    # Load the data
    data = pd.read_csv("stock_data.csv")

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Preprocess the data
    data['Text'] = data['Text'].str.replace('[^\w\s]', '')  # Remove special characters
    data['Text'] = data['Text'].str.lower()  # Convert to lowercase

    # Split the data into training and testing sets
    TRAIN_SIZE = input_number / 100
    #X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Sentiment'], test_size=1-TRAIN_SIZE, random_state=42)

    total_samples = len(data)
    #TRAIN_SIZE_PERCENT = 0.8  # 80% of samples for training
    train_size = int(total_samples * TRAIN_SIZE)
    test_size = int(total_samples * 0.8)

    # Split the data into training and test sets
    train_set = data[:train_size]
    test_set = data[test_size:]

    # Train-test split for text and sentiment columns
    X_train, X_test = train_set['Text'], test_set['Text']
    y_train, y_test = train_set['Sentiment'], test_set['Sentiment']


    # Train the Naive Bayes classifier
    classifier = NaiveBayesClassifier()
    classifier.fit(X_train.str.split(), y_train)  # Convert text to list of words before fitting
    print("Training classifier…")
    print("Testing classifier…")
    # Predict on the test set
    predictions, probabilities = classifier.predict(X_test.str.split())  # Convert text to list of words before predicting


    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, predictions)
    tp, tn, fp, fn = conf_matrix[1, 1], conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0]
    sensitivity = recall_score(y_test, predictions)
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, predictions)
    neg_pred_value = tn / (tn + fn)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Print test results
    print("Test results / metrics:")
    print("Number of true positives:", tp)
    print("Number of true negatives:", tn)
    print("Number of false positives:", fp)
    print("Number of false negatives:", fn)
    print("Sensitivity (recall):", sensitivity)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("Negative predictive value:", neg_pred_value)
    print("Accuracy:", accuracy)
    print("F-score:", f1)
    print()


    # Classification of user input
    while True:
        print("Enter your sentence:")
        input_sentence = input("Sentence S: ")

        # Classify the input sentence
        input_tokenized = input_sentence.lower().replace('[^\w\s]', '').split()
        predicted_sentiment, predicted_probabilities = classifier.predict([input_tokenized])


        # Print the predicted sentiment and probabilities
        print("Predicted Sentiment:", predicted_sentiment[0])
        print("Probabilities:", predicted_probabilities[0])


        # Store probabilities in variables
        positive_probability = predicted_probabilities[0][1]
        negative_probability = predicted_probabilities[0][-1]

        print("Positive Probability:", positive_probability)
        print("Negative Probability:", negative_probability)



        print("was classified as :", predicted_sentiment[0])

        print("P(positive / 1 | %s ) = " % (input_sentence),positive_probability)
        print("P(negative / -1 | %s ) = " % (input_sentence),negative_probability)
        choice = input("Do you want to enter another sentence [Y/N]? ")
        if choice.upper() != 'Y':
            break


        
