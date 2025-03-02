import json
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import PyPDF2
import streamlit as st
from datetime import datetime

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
pdf_path = "ResilienceBOT.pdf"
form_responses_path = "RB01.txt"
conversations_json = "cleaned_data.json"

# Chargement et encodage des documents
conversation_dict, documents = load_data(conversations_json, pdf_path, form_responses_path)
document_embeddings = embedder.encode(documents, batch_size=32, show_progress_bar=True) if documents else []

# ---------------------------------------------------------------------
# 3. Recherche de contexte pertinent via similarité
# ---------------------------------------------------------------------
def retrieve_context(query, strong_threshold=0.6, weak_threshold=0.4):
    try:
        if document_embeddings is None or len(document_embeddings) == 0:
            return None, None

        query_embedding = embedder.encode([query])
        similarities = cosine_similarity(query_embedding, document_embeddings)[0]
        
        if len(similarities) == 0:
            return None, None
            
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        if best_similarity >= strong_threshold:
            return documents[best_match_idx], 'strong'
        elif best_similarity >= weak_threshold:
            return documents[best_match_idx], 'weak'
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération du contexte : {str(e)}")
        return None, None

# ---------------------------------------------------------------------
# 4. Gestion de l'historique de conversation
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None

def add_message(sender, message):
    st.session_state.messages.append({"role": sender, "content": message})

# ---------------------------------------------------------------------
# 5. Fonction de traduction
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
# 6. Génération de réponse améliorée avec contexte conversationnel
# ---------------------------------------------------------------------
def generate_response(user_query: str) -> str:
    # Gestion des salutations
    query_clean = user_query.lower().strip()
    french_greeting = (
        "Bonjour ! Je suis ResilienceBOT, mais tu peux m’appeler RB—ton compagnon personnel sur ce chemin.\n\n"
        "Je ne suis pas humain et je ne suis pas psychologue, mais je suis là pour t’accompagner, t’aider dans ton développement, ta résilience et ton bien-être.\n\n"
        "Tu as des forces de caractère uniques et un potentiel précieux, et j’aimerais les explorer avec toi. Je suis ici pour t’aider à réfléchir et t’accompagner dans tes propres découvertes.\n\n"
        "Si tu ressens le besoin d’un accompagnement plus approfondi, je t’encourage à consulter des ressources ou des professionnels adaptés à ta situation. "
        "Qu’as-tu en tête aujourd’hui ?"
    )
    english_greeting = (
        "Hello! I’m ResilienceBOT, but you can call me RB—your personal companion on this journey.\n\n"
        "I’m not human, and I’m not a psychologist, but I’m here to support, encourage, and guide you as you work on your growth, resilience, and well-being.\n\n"
        "You have unique character strengths and valuable potential, and I’d love to explore them with you. I’m here to help you reflect and accompany you in your own discoveries.\n\n"
        "If you ever feel the need for deeper guidance, I encourage you to seek resources or professionals best suited to your situation. "
        "What’s on your mind today?"
    )
    if query_clean in ["bonjour", "hi"]:
        return french_greeting if query_clean == "bonjour" else english_greeting

    # Détection de langue
    try:
        lang = detect(user_query)
        lang = lang if lang in ["fr", "en"] else "en"
    except:
        lang = "en"

    # Traduction de la requête
    query_en = translate_text(user_query, "French", "English") if lang == "fr" else user_query

    # Vérification de contexte initiale
    context_check, _ = retrieve_context(query_en)
    if not context_check:
        return (
            "Coucou ! Je suis content d'avoir de tes nouvelles. Qu'est-ce qui te passe par la tête aujourd'hui ? J'adorerais discuter de tout ce qui peut t'aider à t'épanouir et à te sentir au mieux."
            "Je te remercie d'avoir pris contact ! Si quelque chose te préoccupe, explorons-le ensemble. Comment s'est passée ta journée jusqu'à présent ?"
        ) if lang == "fr" else (
            "Hey there! I'm happy to hear from you. What's been on your mind today? I'd love to chat about anything that helps you grow and feel your best."
            "I appreciate you reaching out! If something is on your mind, let's explore it together. How has your day been so far?"
        )

    # Construction de l'historique conversationnel
    conversation_history = ""
    if len(st.session_state.messages) >= 2:
        recent_history = st.session_state.messages[-4:]  # 2 derniers tours
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if lang == "fr":
                content_en = translate_text(content, "French", "English") if detect(content) == "fr" else content
                conversation_history += f"{role}: {content_en}\n"
            else:
                conversation_history += f"{role}: {content}\n"

    # Mise à jour du thème conversationnel
    if "presentation" in query_en.lower() or "anxious" in query_en.lower():
        st.session_state.current_topic = "public_speaking_anxiety"
    elif "stress" in query_en.lower() and st.session_state.current_topic:
        query_en += f" (related to {st.session_state.current_topic})"

    # Récupération du contexte final
    context, _ = retrieve_context(f"{conversation_history}\n{query_en}")  # <-- Ligne cruciale ajoutée

    # Construction du prompt contextuel
    prompt = f"""
You are a positive psychology agent, not a human.
You are a positive psychology agent that specializes in Education issues.
Respond concisely and directly to the user's input.
Do not include any internal data or variable names in your answer.
Do not include any references to internal models or context names in the answer.
Use the available psychological and behavioral data to construct a meaningful response.

[User's Character Strengths]
Current strengths:
1. Spirituality and faith
2. Kindness and generosity
3. Social intelligence
4. Perspective wisdom
5. Honesty and authenticity
6. Fairness and equity

Strengths to develop more:
1. Spirituality and faith
2. Fairness and equity
3. Perspective wisdom
4. Social intelligence
5. Kindness and generosity

[Conversation History]
{conversation_history}

[Relevant Information]
{context if context else 'No specific context found'}

[Current Query]
{query_en}

[Response Guidelines]
1. Maintain natural flow and continuity
2. Reference previous exchanges when relevant
3. Provide practical, actionable advice
4. Show empathy and encouragement
5. Connect concepts for deeper understanding
6. Focus on {st.session_state.current_topic if st.session_state.current_topic else 'general well-being'}

Response:
"""

    # Génération de la réponse
    try:
        response = ollama.chat(
            model='minicpm-v',
            messages=[{"role": "user", "content": prompt}]
        )['message']['content'].strip()
        
        if lang == "fr":
            response = translate_text(response, "English", "French")
        
        return response

    except Exception as e:
        st.error(f"Erreur : {e}")
        return "Je rencontre des difficultés techniques. Pouvez-vous reformuler votre question ?"

# ---------------------------------------------------------------------
# 7. Interface Streamlit
# ---------------------------------------------------------------------
st.set_page_config(page_title="ResilienceBOT", page_icon="🤖")
st.title("ResilienceBOT")

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Gestion des interactions
if user_input := st.chat_input("Posez votre question ou décrivez votre situation..."):
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Génération et affichage de la réponse
    with st.spinner("Réflexion en cours..."):
        bot_response = generate_response(user_input)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.write(bot_response)

# Sauvegarde automatique optionnelle
if st.session_state.get('auto_save', False):
    save_conversation()
