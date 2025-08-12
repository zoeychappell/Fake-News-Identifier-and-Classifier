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
import nltk
import pandas
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Download the required Natural Language  Toolkit resources
nltk.download('punkt')
nltk.download('stopwords')

'''
This function takes in the csv file, splits it into messages
and then tokenizes the messages.

Parameters: 
    file : a .csv file
    
Returns: 
    Tokenized messages
'''
def tokenize(file):
    messages = [] # = 9900 
    labels = [] # = 9900

    try: 
        df = pandas.read_csv(file)
    except FileNotFoundError as fnfe:
        print(fnfe)
        return
    
    tokenizer = RegexpTokenizer(r'\w+')
    # pulls out all the message and tokenizes them
    for row in df['Text']:
        tokens = tokenizer.tokenize(str(row))
        messages.append(tokens)
    #pulls out all the labels
    for row in df['label']:
        labels.append(row)
    
    return messages, labels

'''
Takes out all the stopwords in the messages. 
Parameters: 
    messages : the tokenized messages
    
Returns: 
    cleaned_messages = messages minus the stopwords
'''
'''
helper function for clean() and checks for words like 2017Election
Parameters: 
    - token 
Returns: 
    boolean of whether the string matches'''
def word_has_num(token):
    # regex checks for a word that begins with a number. 
    return bool (re.match(r'^\d', token) or re.search(r'\d$', token))
'''
Helper function for clean(), looks for strings like u1rd4b6cz2 or pj9ej1tmw1
Parameters: 
    token 
Returns: 
    boolean of whether the string matches
'''
def word_has_nums_and_letters(token):
    # regex checks for uppercase and lowercase letters, digits, and at least 5 characters long
    return bool (re.match('[a-zA-Z0-9]{5,}', token))
'''
Cleans the message. 
Parameter: 
    message
    
Note: some words im considering removing: 
via, image, video, single letters
instead of reuter's it says reut rs'''
def clean(messages): 
    # custom list of words to be filtered out. 
    custom_unwanted = [# months
                        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 
                        # week days
                        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 
                       # stopwords not removed
                       'said', 'could', 'soon', 'told', 'says', 'also', 'since', 'much', 'like', 'every', 'went', 'made', 'might', 'would', 'puts'
                       # note, this line is like news anchors - could keep if wanted to do something with
                       'cspan', 'fox', 'cn', 'snl'         
                       # random
                       'http', 'https', 'getty', 'ohm', 'lulu', 'lrso', 'sulu', 'oompa', 'jaly', 'abney', 'jpg', 'sdf', 'ubs', 'ch', 'ygd', 'ttp'
                       'kwame', 'shiya', 'boas', 'kiche', 'assa', 'omni', 'jaly', 'pamby', 'acdc', 'gopac', 'tenga',  'ym', 'noy', 'oz', 'da', 'ge'
                       'kassy', 'w1', 'erika', 'cotti', 'df1', 'bpfh', 'fku', 'lir', 'gaier', 'syed', 'dje', 'edva', 'abedi', 'hk', 'zoe', 'eog', 'kok'
                       'zakka', 'karim', 'madi', 'svcs', 'oag', 'ramaj', 'eroc', 'eau' 'haass', 'kteg', 'dubke', 'sergi', 'kirt', 
                       'ddhq', 'kptv', 'gwich','s1ppi', 'josie'
                       ]
    # creates a set of stopwords like 'a', 'the', 'this'
    stop_words = set(stopwords.words('english'))
    cleaned_messages = []
    #iterates through the messages and cleans them
    for message in messages: 
        cleaned_tokens = [token.lower() for token in message 
                          # checks if token in stop_words
                          if token.lower() not in stop_words
                          # checks if token is not in custom_unwanted
                          and token.lower() not in custom_unwanted
                          #checks that token is not a digit 
                          and not token.isdigit()
                          # checks that the word doesn't start with a number, ex. 2017Election
                          and not word_has_num(token)
                          # checks that the word isn't a random string of letters and numbers
                          and not word_has_nums_and_letters(token)
                          # checks that the token is not just a single letter
                          and not len(token.lower())==1
                          # checks to see if underscores in the message
                          and '_' not in token
                          ]
        cleaned_messages.append(cleaned_tokens)
    return cleaned_messages
'''
Uses the sklearn function train_test_split to split the dataset. 

Parameters: 
    messages -> the tokenized dataset
    labels -> the tokenized 
    size -> the percentage of the test set.
'''
def split_dataset(messages, labels, size):
    # test_size -> what percent goes into testing
    # random_state = 42 -> makes the split reproducable
    x_train, x_test, y_train, y_test  = train_test_split(messages, labels, test_size=size, random_state=42)
    return x_train, x_test, y_train, y_test 
'''
Takes in the cleaned messages and returns the TF-IDF feature vector
Parameters: 
    cleaned_messages -> the cleaned, tokenized messages
    max_feature -> the max number of features, will limit the vocab size
Returns: 
    tf_idf_matrix 
    vectorizer
'''
def tf_idf(cleaned_messages, max_feature):
    # create the tfidf vectorizer. n_gram range = unigrams and bigrams
    vectorizer = TfidfVectorizer(max_features = max_feature, ngram_range = (1,2))
    tf_idf_matrix = vectorizer.fit_transform(cleaned_messages)
    return tf_idf_matrix, vectorizer

'''
Creates the feature set of every unique word. 
Parameters: 
    cleaned_message -> the cleaned and tokenized messages
    
Returns: 
    bag_of_words -> the uniue set of words
'''
def bag_of_words(cleaned_messages):
    bag_of_words = []
    # checks each word in each message
    for message in cleaned_messages:
        for token in message:
            # if word not in the list already, adds it
            if token not in bag_of_words:
                bag_of_words.append(token)
    print(bag_of_words)
    return bag_of_words
'''
Applies PCA to reduce the dimensionality of the tfidf vectors
Parameters: 
    tfidf_matrix -> matrix from tf_idf()
    n_components -> number of principal components
Returns: 
    X_pca -> PCA transformed data
    pca -> fitted PCA object
'''
def pca(tfidf_matrix, n_components = 100):
    # tfidf_matrix is sparse (mostly zeros)
    dense_matrix = tfidf_matrix.toarray()
    # applies pca
    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(dense_matrix)
    
    return X_pca, pca
    

def main(): 
    file = "./fake_and_real_news.csv"
    messages, labels = tokenize(file)
    cleaned_messages = clean(messages)
    split_dataset(messages, labels, .2)
    bag_of_words(cleaned_messages)
    

if __name__ == '__main__':
    main()
