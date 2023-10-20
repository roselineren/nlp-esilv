
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




