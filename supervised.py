'''
Zoey Chappell, Luke McEwen, and Daniel Wolosiuk
Saniat Sohrawardi
CSEC 520

AI Usage Statement
Tools Used: ChatGPT
- Usage: Brainstorming suitable libraries. Error detection. Regex format. 
- Verification: Cross-checked with library manual page and manual testing
Prohibited Use Compliance: Confirmed
'''
from data_process import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt


def evaluate_knn_with_varying_features(messages, labels, feature_values):
    """Tests KNN with different max_features values and plots the results
    
    Args:
        messages: List of text messages
        labels: List of corresponding labels
        feature_values: List of max_features values to test
    """
    # Preprocess messages (join if they're tokenized)
    messages = [" ".join(word_list) if isinstance(word_list, list) else str(word_list) 
               for word_list in messages]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    accuracies = []
    
    for max_feat in feature_values:
        # Vectorize with current max_features
        tfidf = TfidfVectorizer(
            lowercase=True,
            max_features=max_feat
        )
        X = tfidf.fit_transform(messages)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"max_features={max_feat}: Accuracy = {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, accuracies, marker='o')
    plt.xlabel('max_features')
    plt.ylabel('Accuracy')
    plt.title('KNN Performance vs. Vocabulary Size')
    plt.grid(True)
    plt.savefig('knn_accuracy.png')
    plt.show()

def do_knn(messages, labels, max_features):
    """Perform K-NN once, print metrics

    Args:
        messages: The total messages that are to be processed 
        labels: The labels parallel to the messages 
        max_features: The maximum number of features (top vocabulary words) to consider
    """
    messages = [" ".join(word_list) for word_list in messages]
    # Convert to TF-IDF
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels) #convert "Real" and "Fake" to 1s and 0s
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        stop_words='english'
    )
    x = tfidf.fit_transform(messages)
    x_train, x_test, y_train, y_test  = split_dataset(x, y, .2)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(classification_report(y_test, y_pred))


def do_naive_bayes(messages, labels):
    """Do naive bayes classification

    Args:
        messages: The messages for which to conduct naive bayes on
        labels: The "Real"/"Fake" labels 
    """
    messages = [" ".join(message) for message in messages]
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english'
    )
    x = tfidf.fit_transform(messages)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels) #convert "Real" and "Fake" to 1s and 0s
    x_train, x_test, y_train, y_test  = split_dataset(x, y, .2)
    nb_classifier = MultinomialNB()
    nb_classifier.fit(x_train, y_train)
    y_pred = nb_classifier.predict(x_test) 
    print(classification_report(y_test, y_pred))

def main(): 
    file = "./fake_and_real_news.csv"
    messages, labels = tokenize(file)
    cleaned_messages = clean(messages)
    bag_of_words(cleaned_messages)

    print("KNN: ")
    do_knn(messages, labels, 100)
    #feature_vals = [100, 500, 1000, 1500, 2000, 2500]
    #evaluate_knn_with_varying_features(messages, labels, feature_vals)
    print("Naive bayes: ")
    do_naive_bayes(messages, labels)


if __name__ == '__main__':
    main()
