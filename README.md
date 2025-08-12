# Fake-News-Identifier-and-Classifier
The objective of the project is to train models through supervised and unsupervised learning methods to determine which approach is the most effective. Supervised machine learning algorithms will use the real/fake labels provided to determine whether an untested news article is real or fake. Unsupervised machine learning will be utilized to classify the data into families that were not originally derived from the provided dataset. Examples of families might include political bias, satire, ideological framing, or sensationalism. This project can be used to provide insight into the most common types of disinformation that are spread and will apply to any text-based media. 

# Dataset used: 
https://www.kaggle.com/datasets/nitishjolly/news-detection-fake-or-real-dataset/data

This dataset consists of news items labeled as either “fake” or “real,” and it can be used to identify false or misleading news (Jolly, N). Out of 9865 total values,  51% of the values are labeled “fake,”  and the other 49% are labeled “real” (Jolly, N). The dataset contains raw, noisy text, such as words ending in a colon, that require preprocessing before machine learning algorithms are applied. 

This dataset was created by “Kaggle Expert” Nitish Jolly who is a student at Thapar Institute of Engineering and Technology. 

# Methodology 
## 1. Clean data. 
Each entry in the database is a full-text news article and a label, which was used during the supervised learning algorithms (Jolly, N). 51% of the articles are labeled real and 49% are labeled fake. The nearly equivalent distribution mitigates concerns related to class imbalance that introduce bias to the algorithms. However, the dataset was initially unprocessed and required extensive cleaning before being suitable for use in machine learning algorithms. 

A key component of our approach is a custom data processing function, in which a mix-and-match approach can be taken to the data cleaning process. This is essential for preparing the dataset for the machine learning algorithms.

The data cleaning process proceeds as follows:
1. All tokens are converted to lowercase to mitigate case-sensitivity issues.
2. Common stop words are removed using NLTK’s built-in list of English stop words. Note, stop words are
widely used words such as “the” or “in” (Web Communications, 2025).
3. Filter the tokens, based on:
  1. Exact matches to words in the custom unwanted list.
  2. Tokens that are numeric or contain letters and numbers (e.g., 2017 Election).
  3. Tokens that include underscores.
  4. Tokens that match random alphanumeric strings (e.g. u1rd4b6cz2) identified using regular expressions.

## 2. Transform Data: 
The tokenized data is converted into feature vectors using the Term Frequency - Inverse Document Frequency (TF-IDF) method. TfidfVectorizer from scikit-learn was configured to extract both unigrams and bigrams, with an adjustable max_features parameter to control vocabulary size. 
Principal Component Analysis (PCA) was applied to reduce the dimensionality of the sparse TF-IDF matrix, resulting in an improved interpretability of the data while retaining the most significant parts. 
Finally, the data was split into training and testing sets using the train_test_split() function from scikit-learn. A fixed random seed was used to ensure a reducible split. 

## 3. Apply supervised learning algorithms to classify data. 
The objective of supervised learning is accurate classification of untested fake news articles. 

K-NN was considered for its non-linear decision boundaries and and interpretability. Non-linear decision boundaries are valuable because complex patterns can be observed when combined with well structured feature space. Interpretability is considered because "nearest neighbors" is very useful for explaining why a prediction decision was made.

Naive Bayes was primarily considered for its resistance to overfitting data. Independence assumptions are present in this model, which keeps the model "simple". After training, Naive Bayes can very quickly predict labels on untested data. This is very useful for application-level implementations, such as content filtering systems.

Both models feature a configurable "vocabulary size" parameter. This parameter selects the "top x most important words" of a given text and uses only those for consideration.

## 4. Apply unsupervised learning algorithms to label groups.
The goal of the unsupervised learning portion is to attempt to discover groupings of news articles without the need to label. 

K-Means was picked for its simplicity, speed, and interpretability. K-Means is efficient at partitioning data into distinct clusters making a good baseline. K-Means was implemented with a variable number of clusters which looped from k=2 to k=20. For each value the clustering output was compared against the true labels using metrics. These scores were plotted to visualize performance variance between the range of numbers.

DBSCAN was chosen for its ability find outliers offering a complementary perspective to K-Means. DBSCAN does not require a specific number of clusters but instead relies on two key parameters, neighborhood radius (eps), and minimum points to form a cluster (min\_samples). The eps parameter was again varied from 0.3 to 1.2 and then scores of the clustering were plotted and compared to see which value of epsilon has the best success.

Both used a set of vectorized articles using TF-IDF, followed by dimensionality reduction with PCA.  

