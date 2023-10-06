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
import os

# %%
# Chemin vers le dossier où vous avez décompressé le dataset
dataset_directory = "C:/Users/rosel/Desktop/ML_NLP/txt_sentoken"

# %%
# Sous-répertoires pour critiques positives et négatives
pos_dir = os.path.join(dataset_directory, 'pos')
neg_dir = os.path.join(dataset_directory, 'neg')

# %%
def load_reviews(directory, label):
    """Charge les critiques d'un répertoire donné et attribue une étiquette"""
    reviews = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read().strip()
            reviews.append((text, label))
    return reviews

# %%
# Charger les critiques positives et négatives
positive_reviews = load_reviews(pos_dir, 'positive')
negative_reviews = load_reviews(neg_dir, 'negative')


# %%
# Combiner les critiques
all_reviews = positive_reviews + negative_reviews

# %% [markdown]
# Mélanger le dataset: pour plus tardl'entrainer et le test

# %%
import random

random.shuffle(all_reviews)

# %%
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Tokenise et étiquette morpho-syntaxique pour chaque critique
tagged_reviews = [(word_tokenize(review), label) for review, label in all_reviews]
tagged_reviews = [(nltk.pos_tag(tokens), label) for tokens, label in tagged_reviews]

# %%
def extract_adverbs(tagged_tokens):
    """Extrait les adverbes d'une liste de tokens étiquetés."""
    return [word for word, pos in tagged_tokens if pos.startswith('RB')]

# Extraire les adverbes pour chaque critique
adverbs_in_reviews = [(extract_adverbs(tagged_tokens), label) for tagged_tokens, label in tagged_reviews]

# %%
# adverbs_in_reviews

# %%
from nltk.corpus import sentiwordnet as swn

def get_sentiment(adverb):
    """Obtient le score de sentiment pour un adverbe à l'aide de SentiWordNet."""
    synsets = list(swn.senti_synsets(adverb, 'r'))  # 'r' pour adverbes
    if not synsets:
        return 0  # Aucun score si l'adverbe n'est pas trouvé dans SentiWordNet
    
    # Utiliser le premier synset par défaut (pourrait être amélioré en utilisant des méthodes de désambiguïsation)
    return synsets[0].pos_score() - synsets[0].neg_score()

# Calculer le score de sentiment pour chaque adverbe dans les critiques
sentiments_in_reviews = [(sum(get_sentiment(adverb) for adverb in adverbs), label) for adverbs, label in adverbs_in_reviews]

# %%
# sentiments_in_reviews

# %%
#Classer les critiques en fonction des scores de sentiment
def classify_review(sum_score):
    return "pos" if sum_score > 0 else "neg"

predicted_labels = [classify_review(score) for score, _ in sentiments_in_reviews]

# Calculer la précision de la classification
actual_labels = [label for _, label in sentiments_in_reviews]
correctly_classified = sum(1 for predicted, actual in zip(predicted_labels, actual_labels) if predicted == actual)

accuracy = correctly_classified / len(predicted_labels)

print(f"Précision de la classification: {accuracy * 100:.2f}%")

# %%
X = [sum_ for sum_, _ in sentiments_in_reviews]
y = [label for _, label in sentiments_in_reviews]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression

# Reshape les données car nous avons une seule caractéristique
X_train = [[x] for x in X_train]
X_test = [[x] for x in X_test]

# Entraîner le modèle
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score, classification_report

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# %% [markdown]
# Votre modèle a une précision de 47%, ce qui n'est pas idéal, essayons de l'améliorer

# %% [markdown]
# TF-IDF (Term Frequency-Inverse Document Frequency):
# 
# Au lieu de se concentrer uniquement sur le score de sentiment, nous pouvons transformer les critiques en vecteurs numériques à l'aide de TF-IDF. Cette technique prend en compte l'importance d'un mot dans un document par rapport à l'ensemble du corpus.

# %%
print(all_reviews[:5])  # Affichez les 5 premières entrées pour vérifier.

# %% [markdown]
# D'accord, il semble que all_reviews soit une liste de tuples, où chaque tuple contient une seule chaîne de caractères (la critique).

# %%
all_reviews = [review[0] for review in all_reviews]

# %% [markdown]
# Cela transformera all_reviews en une simple liste de chaînes de caractères. 

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Créez un vecteur TF-IDF basé sur les critiques
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(all_reviews)

# Divisez à nouveau les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# %%
import streamlit as st
import joblib

# %%
# Enregistrez le modèle formé et le vectorizer pour une utilisation ultérieure
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
    prediction = model.predict([processed_text])
    
    # Afficher le résultat
    if prediction[0] == 1:  # Supposons que 1 soit pour "positif"
        st.write("Le texte semble avoir un sentiment positif.")
    else:  # Supposons que 0 soit pour "négatif"
        st.write("Le texte semble avoir un sentiment négatif.")



# %%


# %%



