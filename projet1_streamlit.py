import streamlit as st
import pandas as pd
import random
from transformers import BertTokenizer, BertModel
import torch

import urllib.request as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

import nltk
from sklearn.metrics import ndcg_score
from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')

nb_docs=150

def loadNFCorpus():
	#dir = "C:/Users/rosel/OneDrive/Documents/GitHub/nlp-esilv/"
	filename = "dev.docs"

	dicDoc={}
	with open(filename,encoding='utf-8') as file:
		lines = file.readlines()
	for line in lines:
		tabLine = line.split('\t')
		#print(tabLine)
		key = tabLine[0]
		value = tabLine[1]
		#print(value)
		dicDoc[key] = value
	filename = "dev.all.queries"
	dicReq={}
	with open(filename,encoding='utf-8') as file:
		lines = file.readlines()
	for line in lines:
		tabLine = line.split('\t')
		key = tabLine[0]
		value = tabLine[1]
		dicReq[key] = value
	filename = "dev.2-1-0.qrel"
	dicReqDoc=defaultdict(dict)
	with open(filename, encoding='utf-8') as file:
		lines = file.readlines()
	for line in lines:
		tabLine = line.strip().split('\t')
		req = tabLine[0]
		doc = tabLine[2]
		score = int(tabLine[3])
		dicReqDoc[req][doc]=score

	return dicDoc, dicReq, dicReqDoc

# Charger les données depuis le fichier de questions
def load_questions(filename):
    # Assumant que le fichier est un fichier texte avec un ID et une question par ligne, séparés par des tabulations
    data = pd.read_csv(filename, sep='\t', header=None, names=['id', 'question'])
    return data

# Sélectionner une question aléatoire
def get_random_question(data):
    return data.sample().iloc[0]['question']

# Nom du modèle BiomedBERT
med_bert_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

# Chargement du tokenizer et du modèle
tokenizer_med = BertTokenizer.from_pretrained(med_bert_model)
model_med = BertModel.from_pretrained(med_bert_model)

# Fonction pour obtenir les embeddings BERT d'un texte
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pool the token embeddings
    return embeddings

def get_top_documents_for_question(question, dic_doc, model, tokenizer, top_k=5):
    model.eval()

    # Préparer les embeddings pour les documents
    docs_to_keep = list(dic_doc.keys())[:nb_docs]
    corpus_doc_vectors = [get_bert_embeddings(dic_doc[doc_id], model, tokenizer) for doc_id in docs_to_keep]
    corpus_doc_vectors = torch.cat(corpus_doc_vectors, dim=0)

    # Obtenir les embeddings pour la question
    question_embedding = get_bert_embeddings(question, model, tokenizer)

    # Calculer les scores de similarité
    doc_scores = torch.matmul(corpus_doc_vectors, question_embedding.t()).squeeze(1).cpu().numpy()

    # Trier et sélectionner les top documents
    ranked_docs = sorted(zip(docs_to_keep, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked_docs

# Créer l'application Streamlit
def main():
    st.title('Générateur de Questions Aléatoires avec BiomedBERT (la meilleure approche)')

    # Charger les questions
    questions = load_questions('dev.all.queries') 

    # Bouton pour générer une nouvelle question
    if st.button('Générer une Question Aléatoire'):
        random_question = get_random_question(questions)
        st.write(random_question)
        
        # Charger les données NFCorpus globalement
        dic_doc, dic_req, dic_req_doc = loadNFCorpus()
        # Obtenir les top documents pour la question
        top_documents = get_top_documents_for_question(random_question, dic_doc, model_med, tokenizer_med, top_k=5)

        st.write("Top 5 des documents pertinents :")
        for doc_id, score in top_documents:
            st.write(f"ID: {doc_id}, Score: {score}")
            st.text(f"Document: {dic_doc[doc_id]}")

if __name__ == "__main__":
    main()
