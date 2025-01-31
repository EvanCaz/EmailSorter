import re
import os
import math
import nltk
import numpy as np # for dop prodoct
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')
        
stopWords = { 'a', 'an', 'the', 'and', 'or', 'but', 'to', 'of', 'for', 'in', 'on', 'at', 'by', 'with', 'from', 'your', 'we', 'our', 'is', 'that', 'like', 'you', 
             'move', 'have', 'this', 'can', 'still', 'whom', 'other', 'about', 'wish', 'make', 'will', 'eye', 'after', 'when', 'do', 'out', 'dear' }
## nltk removes stop words i like, so i use my own list
    
def getWordBase(tag): # added this because the calculates were off and i thught lemmy alr did this
    return {
        'N' : wordnet.NOUN,
        'V' : wordnet.VERB,
        'J' : wordnet.ADJ,
        'R' : wordnet.ADV
    }.get(tag[0], wordnet.NOUN) # i think most are nouns maybe
    
def processText(rawText):
    rawText = re.sub(r'[^a-zA-Z\'\s]', '', rawText) # remove stuff like special characters
    rawText = re.sub(r'http\S+', '', rawText) # remove url
    tokens = pos_tag(re.findall(r"\b(?:[a-z]+(?:['’-]?[a-z]+)*)+\b", rawText.lower())) 
    lemmy = WordNetLemmatizer()
    lemmatized = []
    for word, tag in tokens:
        pos = getWordBase(tag)
        lemma = lemmy.lemmatize(word, pos=pos)
        lemmatized.append(lemma)
    return [token for token in lemmatized if token not in stopWords and len(token) > 2]
    
def normalize(vector):
    normalization = np.linalg.norm(vector)
    return vector / normalization if normalization != 0 else vector

def cosineSim(a, b):
    return np.dot(a, b) # love np

uniqueVocabulary = {} # for idf vecotrization, keys are the unique words, which are string, and the vlaues are an integer which represents number of occurances across all documents
documentTF = []
path = r"/mnt/c/Users/Schmoopy/Documents/EmailSorter/emails" #ubunto not cmd
boostWords = {"invite": 5, "not": 5}
# boostWords = {}
# boostWords = {"not": 3}

for filename in os.listdir(path): # need to calculate tf, frequency in an email, and idf, frequency across all emails
    file_path = os.path.join(path, filename)
    with open(file_path, "r") as file:
        rawText = file.read()
    text = processText(rawText)
    
    wordCounts = {} # dict for tf
    totalWord = len(text)
    for word in text:
        increment = boostWords.get(word, 1.0)
        wordCounts[word] = wordCounts.get(word, 0) + increment # like that leetcode question 2225 i think
    tfCur = {word: count / totalWord for word, count in wordCounts.items()}
    
    documentTF.append(tfCur) # each documents respective tf scores for each word
    
    for word in set(text): # so we only count each word once, right now with three docuemnts, the total number a word can appear is three and I had more before set but do not use this for tf as frequency matters
            if word in uniqueVocabulary:
                uniqueVocabulary[word] += 1  
            else:
                uniqueVocabulary[word] = 1  

# print(uniqueVocabulary) #testing
# the above two nested for loops can be combined, just not yet
uniqueVocabularyIDF = {word: math.log(len(os.listdir(path)) / (df + 1)) for word, df in uniqueVocabulary.items()}

# fcalculate a score for each word in the document with the idf scroe from dict and tf score of cur documenet, append all of them to a list, and append that list to list of all vectors 
vocab = sorted(uniqueVocabulary.keys())
tfidfVector = []
for i in documentTF: 
    vector = []
    for word in vocab:
        tfidfScore = i.get(word, 0.0) * uniqueVocabularyIDF.get(word, 0.0)
        vector.append(tfidfScore)
    tfidfVector.append(vector)

normalized_vectors = [normalize(v) for v in tfidfVector] # each index is a email
# all vectors must be scalled by one because thats how cosine similarity works
    
for i in range(len(normalized_vectors)):
    for j in range(i + 1, len(normalized_vectors)):  
        similarity = cosineSim(normalized_vectors[i], normalized_vectors[j])
        print(f"Similarity between email {i + 1} and email {j + 1}: {100 * similarity:.8f}")