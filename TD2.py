# %% [markdown]
# # TD2: Parts of Speech tagging for sentimment analysis

# %% [markdown]
# Part-of-speech tagging is the process of converting a sentence, in the form of a list of words,
# into a list of tuples, where each tuple is of the form (word, tag). The tag is a part-of-speech
# tag, and signifies whether the word is a noun, adjective, verb, and so on.
# 
# Most of the taggers are trainable. They use a list of tagged sentences as their training data, such as
# what you get from the tagged_sents() method of a TaggedCorpusReader class. With these training
# sentences, the tagger generates an internal model that will tell it how to tag a word. Other taggers
# use external data sources or match word patterns to choose a tag for a word.
# All taggers in NLTK are in the nltk.tag package. Many taggers can also be combined into a backoff
# chain, so that if one tagger cannot tag a word, the next tagger is used, and so on.

# %% [markdown]
# Training a unigram part-of-speech tagger
# 
# UnigramTagger can be trained by giving it a list of tagged sentences at initialization.
# 
# >>> from nltk.tag import UnigramTagger
# 
# >>> from nltk.corpus import treebank
# 
# >>> train_sents = treebank.tagged_sents()[:3000]
# 
# >>> tagger = UnigramTagger(train_sents)
# 
# >>> treebank.sents()[0]
# 
# ['Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director','Nov.', '29', '.']
# 
# >>> tagger.tag(treebank.sents()[0])
# 
# [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will',
# 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'),('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]

# %% [markdown]
# We use the first 3000 tagged sentences of the treebank corpus as the training set to
# initialize the UnigramTagger class. Then, we see the first sentence as a list of words,
# and can see how it is transformed by the tag() function into a list of tagged tokens.

# %%
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')


# %%
import streamlit as st
import joblib

# %%
# Enregistrez le modèle formé et le vectorizer pour une utilisation ultérieure
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)
joblib.dump(classifier, 'c:/Users/rosel/Desktop/ML_NLP/sentiment_model.pkl')
joblib.dump(vectorizer, 'c:/Users/rosel/Desktop/ML_NLP/tfidf_vectorizer.pkl')

# %%
# Charger le modèle pré-entraîné
model = joblib.load('c:/Users/rosel/Desktop/ML_NLP/sentiment_model.pkl')

# %%
st.title('Analyseur de sentiment')


# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# %%
def preprocess(text):
    # Convertir le texte en minuscules
    text = text.lower()
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Suppression des mots vides
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# %%
# Créer une boîte de texte pour l'entrée utilisateur
user_input = st.text_area("Entrez votre texte ici")

if st.button('Analyser'):
    processed_text = preprocess(user_input)
    
    # Transformez le texte prétraité avec le vecteur TF-IDF
    X_user_input = vectorizer.transform([processed_text]) 
    
    # Prédire le sentiment
    prediction = model.predict(X_user_input)
    
    # Afficher le résultat
    if prediction[0] == 1:  # Supposons que 1 soit pour "positif"
        st.write("Le texte semble avoir un sentiment positif.")
    else:  # Supposons que 0 soit pour "négatif"
        st.write("Le texte semble avoir un sentiment négatif.")




