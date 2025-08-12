# Fake-News-Identifier-and-Classifier
The objective of the project is to train models through supervised and unsupervised learning methods to determine which approach is the most effective. Supervised machine learning algorithms will use the real/fake labels provided to determine whether an untested news article is real or fake. Unsupervised machine learning will be utilized to classify the data into families that were not originally derived from the provided dataset. Examples of families might include political bias, satire, ideological framing, or sensationalism. This project can be used to provide insight into the most common types of disinformation that are spread and will apply to any text-based media. 

# Dataset used: 
https://www.kaggle.com/datasets/nitishjolly/news-detection-fake-or-real-dataset/data

This dataset consists of news items labeled as either “fake” or “real,” and it can be used to identify false or misleading news (Jolly, N). Out of 9865 total values,  51% of the values are labeled “fake,”  and the other 49% are labeled “real” (Jolly, N). The dataset contains raw, noisy text, such as words ending in a colon, that require preprocessing before machine learning algorithms are applied. 

This dataset was created by “Kaggle Expert” Nitish Jolly who is a student at Thapar Institute of Engineering and Technology. 

# Methodology 
## 1. Clean data. 
Each entry in the database is a full-text news article and a label, which was used during the supervised learning algorithms (Jolly, N). 51% of the articles are labeled real and 49% are labeled fake. The nearly equivalent distribution mitigates concerns related to class imbalance that introduce bias to the algorithms. However, the dataset was initially unprocessed and required extensive cleaning before being suitable for use in machine learning algorithms. A key component of our approach is a custom data processing function, in which a mix-and-match approach can be taken to the data cleaning process. This is essential for preparing the dataset for the machine learning algorithms.
The data cleaning process proceeds as follows:
1. All tokens are converted to lowercase to mitigate case-sensitivity issues.
2. Common stop words are removed using NLTK’s built-in list of English stop words. Note, stop words are
widely used words such as “the” or “in” (Web Communications, 2025).
3. Filter the tokens, based on:
(a) Exact matches to words in the custom unwanted list.
(b) Tokens that are numeric or contain letters and numbers (e.g., 2017 Election).
(c) Tokens that include underscores.
(d) Tokens that match random alphanumeric strings (e.g. u1rd4b6cz2) identified using regular expressions.


