# Submitted by Saurabh Singh - 1001568347
# References:
# 1) https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# 2) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

import pandas as pd
import sys, getopt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# Parse the command line arguments and call respective methods
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ['mode=', 'input='])
        mode = opts[0][1]
        input = opts[1][1]
    except getopt.GetoptError:
        print('Usage: python P2.py --mode <mode> --input <input_file>')
        sys.exit(2)
    if mode == "train":
        train(mode, input)
    elif mode == "cross_val":
        train(mode, input)
    elif mode == "predict":
        predict(input)
    else:
        print('Mode can either be train OR cross_val OR predict.')

def train(mode, input_file):
    voting_data = pd.read_csv(input_file)
    categories = ['E', 'V', 'O', 'Others', 'All']

    # Split data into train and test. By default the method splits in the ratio 3:4. So not mentioned it specifically.
    train_feature, test_feature, train_class, test_class = train_test_split(
        voting_data['text'], voting_data['label'], stratify=voting_data['label'])
    X_train_counts = count_vect.fit_transform(train_feature)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Create a model using LinearSVC
    text_clf = LinearSVC().fit(X_train_tfidf, train_class)

    # Saving to a pickle
    joblib.dump([text_clf, train_feature], 'clf.pkl')

    # Transformation & prediction for test data
    X_new_counts = count_vect.transform(test_feature)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = text_clf.predict(X_new_tfidf)

    # Print the Classification report and Confusion matrix for train mode
    if mode == "train":
        print("Classification Report:\n\n", classification_report(test_class, predicted, target_names=categories))
        print("Confusion Matrix:\n\n", pd.crosstab(test_class, predicted, rownames=['True'], colnames=['Predicted'], margins=True))

    # K-fold cross validation for cross_val mode
    if mode == "cross_val":
        scores = cross_val_score(text_clf, count_vect.fit_transform(voting_data['text']), voting_data['label'], cv=10)
        print("Cross-validation scores: {}".format(scores))
        print("Average cross-validation score: {:.2f}".format(scores.mean()))

# Predict the class label for input statement using the model saved as pickle object
def predict(input):
    # load a pickle object
    dump = joblib.load('clf.pkl')
    trained_model = dump[0]
    X_train = dump[1]
    # Initialize the transformer with vocabulary
    count_vect.fit_transform(X_train)
    predicted = trained_model.predict(count_vect.transform([input]))
    print(predicted[0])

if __name__ == "__main__":
    main(sys.argv[1:])