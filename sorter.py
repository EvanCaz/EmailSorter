import re
import os
import math
import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')

stopWords = { 'a', 'an', 'the', 'and', 'or', 'but', 'to', 'of', 'for', 'in', 'on', 'at', 'by', 'with', 'from', 'your', 'we', 'our', 'is', 'that', 'like', 'you', 'move', 'have', 'this', 'can', 'still', 'whom', 'other', 'about', 'wish', 'make', 'will', 'eye', 'after', 'when' }
## nltk removes stop words i like, so i use my own list
def processText(rawText):
    rawText = re.sub(r'[^a-zA-Z\'\s]', '', rawText)
    tokens = re.findall(r"\b(?:[a-z]+(?:['’-][a-z]+)*)+\b", rawText.lower())
    lemmy = WordNetLemmatizer()
    return [lemmy.lemmatize(token) for token in tokens if token not in stopWords and len(token) > 2]
    
uniqueVocabulary = {} # for idf vecotrization, keys are the unique words, which are string, and the vlaues are an integer which represents number of occurances across all documents
documentTF = []
    
path = r"/mnt/c/Users/Schmoopy/Documents/EmailSorter/emails" #ubunto not cmd
for filename in os.listdir(path): # need to calculate tf, frequency in an email, and idf, frequency across all emails
    file_path = os.path.join(path, filename)
    with open(file_path, "r") as file:
        rawText = file.read()
    text = processText(rawText)
    
    wordCounts = {} # dict for tf
    totalWord = len(text)
    for word in text:
        wordCounts[word] = wordCounts.get(word, 0) + 1 # like that leetcode question 2225 i think
    tfCur = {word: count / totalWord for word, count in wordCounts.items()}
    documentTF.append(tfCur) # each documents respective tf scores for each word
    
    for word in set(text): # so we only count each word once, right now with three docuemnts, the total number a word can appear is three and I had more before set but do not use this for tf as frequency matters
            if word in uniqueVocabulary:
                uniqueVocabulary[word] += 1  
            else:
                uniqueVocabulary[word] = 1  

# the above two nested for loops can be combined, just not yet
vocab = sorted(uniqueVocabulary.keys())
uniqueVocabularyIDF = {word: math.log(len(os.listdir(path)) / (df + 1)) for word, df in uniqueVocabulary.items()}


# print(f"Unique words collected from all files: {documentTF}")
# for i in range(len(documentTF)):
#     print(documentTF[i])
#     print()


