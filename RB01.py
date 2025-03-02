import json
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import PyPDF2
import streamlit as st

# ---------------------------------------------------------------------
# 1. Chargement et préparation des données
# ---------------------------------------------------------------------
def load_pdf_data(pdf_path):
    """Extrait le texte d'un fichier PDF."""
    pdf_text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
    return pdf_text

def load_form_responses(form_responses_path):
    """Charge le contenu d'un fichier texte contenant des réponses d'un formulaire."""
    try:
        with open(form_responses_path, "r", encoding="utf-8") as f:
            form_text = f.read()
        return form_text
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier de réponses du formulaire : {e}")
        return ""

def load_data(conversations_path, pdf_path=None, form_responses_path=None):
    """
    Charge les exemples de conversation depuis un fichier JSON,
    et combine avec les documents issus du PDF et du fichier de réponses.
    """
    with open(conversations_path, encoding="utf-8") as f:
        conversations = json.load(f)
    
    conversation_dict = {
        c['human_value'].lower().strip().replace("'", "").replace(".", ""): c['gpt_value']
        for c in conversations
    }

    documents = [c['gpt_value'] for c in conversations]

    if pdf_path:
        pdf_text = load_pdf_data(pdf_path)
        if pdf_text:
            documents.append("Support PDF: " + pdf_text)

    if form_responses_path:
        form_text = load_form_responses(form_responses_path)
        if form_text:
            documents.append("Réponses du formulaire: " + form_text)

    return conversation_dict, documents

# ---------------------------------------------------------------------
# 2. Initialisation du modèle d'embeddings
# ---------------------------------------------------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Définition des chemins des fichiers directement dans le code
pdf_path = "ResilienceBOT.pdf"       # Remplacez par le chemin de votre PDF
form_responses_path = "RB01.txt"       # Remplacez par le chemin de votre fichier texte ou laissez None
conversations_json = "cleaned_data.json"  # Votre fichier JSON d'exemples

# Chargement et encodage des documents
conversation_dict, documents = load_data(conversations_json, pdf_path, form_responses_path)
document_embeddings = embedder.encode(documents, batch_size=32, show_progress_bar=True) if documents else []

# ---------------------------------------------------------------------
# 3. Recherche de contexte pertinent via similarité
# ---------------------------------------------------------------------
def retrieve_context(query, strong_threshold=0.6, weak_threshold=0.4):
    if document_embeddings is None or len(document_embeddings) == 0:
        return None, None

    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]

    if best_similarity >= strong_threshold:
        return documents[best_match_idx], 'strong'
    elif best_similarity >= weak_threshold:
        return documents[best_match_idx], 'weak'
    else:
        return None, None

# ---------------------------------------------------------------------
# 4. Gestion de l'historique de conversation dans st.session_state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    # On part d'une liste vide afin que le premier message soit celui de l'utilisateur
    st.session_state.messages = []

def add_message(sender, message):
    st.session_state.messages.append({"role": sender, "content": message})

# ---------------------------------------------------------------------
# 5. Fonction de traduction (via Ollama, modèle minicpm-v)
# ---------------------------------------------------------------------
def translate_text(text, source_lang, target_lang):
    translation_prompt = f"Please translate the following text from {source_lang} to {target_lang}:\n\n{text}"
    try:
        translation_response = ollama.chat(
            model='minicpm-v',
            messages=[{'role': 'user', 'content': translation_prompt}]
        )
        return translation_response['message']['content'].strip()
    except Exception as e:
        st.error(f"Erreur lors de la traduction : {e}")
        return text

# ---------------------------------------------------------------------
# 6. Génération de la réponse
# ---------------------------------------------------------------------
def generate_response(user_query: str) -> str:
    """
    1) Détecte la langue de la requête
    2) Vérifie dans conversation_dict
    3) Recherche de contexte dans documents
    4) Construit le prompt (sans exposer les noms de variables internes)
    5) Appelle le modèle via Ollama
    6) Retourne la réponse
    """

    # --- Gestion spécifique des salutations ---
    # On vérifie avant toute transformation si l'utilisateur saisit "bonjour" ou "hi"
    query_clean = user_query.lower().strip()
    french_greeting = (
        "Bonjour ! Je suis ResilienceBOT, mais tu peux m’appeler RB—ton compagnon personnel sur ce chemin.\n\n"
        "Je ne suis pas humain et je ne suis pas psychologue, mais je suis là pour t’accompagner, t’accompagner, t’aider dans ton développement, ta résilience et ton bien-être.\n\n"
        "Tu as des forces de caractère uniques et un potentiel précieux, et j’aimerais les explorer avec toi. Je suis ici pour t’aider à réfléchir et t’accompagner dans tes propres découvertes.\n\n"
        "Si tu ressens le besoin d’un accompagnement plus approfondi, je t’encourage à consulter des ressources ou des professionnels adaptés à ta situation."
        "Qu’as-tu en tête aujourd’hui ?"
    )
    english_greeting = (
        "Hello! I’m ResilienceBOT, but you can call me RB—your personal companion on this journey.\n\n"
        "I’m not human, and I’m not a psychologist, but I’m here to support, encourage, and guide you as you work on your growth, resilience, and well-being.\n\n"
        "You have unique character strengths and valuable potential, and I’d love to explore them with you. I’m here to help you reflect and accompany you in your own discoveries.\n\n"
        "If you ever feel the need for deeper guidance, I encourage you to seek resources or professionals best suited to your situation."
        "What’s on your mind today?"
    )
    if query_clean == "bonjour":
        return french_greeting
    if query_clean == "hi":
        return english_greeting
    # --- Fin gestion spécifique des salutations ---

    # 1. Détection de la langue
    try:
        lang = detect(user_query)
        if lang not in ["fr", "en"]:
            lang = "en"
    except:
        lang = "en"

    # 2. Traduire la question en anglais si nécessaire
    query_en = user_query
    if lang == "fr":
        query_en = translate_text(user_query, "French", "English")

    # Normalisation
    normalized_query = query_en.lower().strip().replace("'", "").replace(".", "")

    # 3. Vérifier le dictionnaire de conversation
    for key, value in conversation_dict.items():
        normalized_key = key.lower().strip().replace("'", "").replace(".", "")
        if normalized_query in normalized_key or normalized_key in normalized_query:
            # Si la réponse doit être en français, retraduisez-la si nécessaire
            if lang == "fr":
                value = translate_text(value, "French", "English")
            return value

    # 4. Recherche du contexte dans documents
    context, _ = retrieve_context(query_en)
    if not context:
        return ("Je suis désolé, je ne peux répondre qu'aux questions en lien avec notre contexte de psychologie positive. "
                "Pourriez-vous reformuler votre question ?")

    # 5. Construction du prompt en anglais sans révéler les noms internes
    prompt = f"""
You are a positive psychology agent, not a human.
Respond concisely and directly to the user's input.
Do not include any internal data or variable names in your answer.
Do not include any references to internal models or context names in the answer.
Use the available psychological and behavioral data to construct a meaningful response.

Relevant Information: {context}

User: {query_en}
Assistant:
"""

    try:
        response_data = ollama.chat(
            model='minicpm-v',
            messages=[{"role": "user", "content": prompt}]
        )
        raw_response = response_data["message"]["content"].strip()
    except Exception as e:
        st.error(f"Erreur lors de l'appel au modèle Ollama : {e}")
        return "Une erreur est survenue lors de l'appel au modèle."

    # 6. Traduire la réponse en français si la question initiale était en FR
    if lang == "fr":
        raw_response = translate_text(raw_response, "English", "French")

    return raw_response

# ---------------------------------------------------------------------
# 7. Interface Streamlit façon "chat" avec st.chat_message et st.chat_input
# ---------------------------------------------------------------------
st.set_page_config(page_title="Chatbot - Positive Psychology", page_icon="🤖")
st.title("ResilienceBOT")

# Affichage de l'historique de la conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Zone de saisie en bas
if user_input := st.chat_input("Posez votre question…"):
    # Ajout du message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Génération de la réponse
    bot_response = generate_response(user_input)

    # Ajout et affichage de la réponse du bot
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)
