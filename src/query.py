import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# 1. Configuración inicial
load_dotenv()
client = OpenAI()

def load_index(path):
    """Carga nuestra 'base de datos' de chunks y embeddings."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embedding(text):
    """Convierte un texto (la pregunta) en un vector."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def vector_search(query_embedding, index_data, top_k=3):
    """
    Busca los fragmentos más similares.
    Técnica: k-NN (k-Nearest Neighbors) usando Similitud Coseno.
    """
    # Extraemos solo los vectores de nuestra base de datos
    chunk_embeddings = [item["embedding"] for item in index_data]
    
    # Calculamos la similitud entre la pregunta y todos los chunks
    # Usamos cosine_similarity de sklearn
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Obtenemos los índices de los 'top_k' más parecidos (los puntajes más altos)
    related_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in related_indices:
        results.append({
            "text": index_data[idx]["text"],
            "score": float(similarities[idx])
        })
    return results

def generate_answer(question, relevant_chunks):
    """Genera la respuesta final usando el contexto recuperado."""
    
    # Unimos los textos de los chunks para pasarlos al prompt
    context = "\n\n".join([c["text"] for c in relevant_chunks])
    
    prompt = f"""
    Eres un asistente de Recursos Humanos de una empresa SaaS. 
    Usa la siguiente información de las políticas internas para responder la pregunta del usuario.
    Si la respuesta no está en el contexto, di que no lo sabes.

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    RESPUESTA:
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0 # Temperatura 0 para que sea preciso y no invente
    )
    return response.choices[0].message.content

def evaluate_response(question, answer, relevant_chunks):
    """
    Agente Evaluador: Analiza la calidad de la respuesta del RAG.
    """
    context = "\n\n".join([c for c in relevant_chunks])
    
    prompt = f"""
    Eres un auditor de calidad de IA. Tu trabajo es evaluar la respuesta de un chatbot de HR.
    
    CRITERIOS DE EVALUACIÓN:
    1. Relevancia: ¿La respuesta responde directamente a la pregunta?
    2. Precisión: ¿La respuesta se basa exclusivamente en los chunks proporcionados?
    3. Completitud: ¿Se usó toda la información importante del contexto?

    DATOS:
    - Pregunta del usuario: {question}
    - Respuesta del sistema: {answer}
    - Contexto recuperado (chunks): {context}

    DEBES RESPONDER ÚNICAMENTE EN FORMATO JSON con estas claves:
    "score": (un número entero de 0 a 10)
    "reason": (una explicación de al menos 50 caracteres justificando el puntaje)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={ "type": "json_object" } # Forzamos a que devuelva JSON
    )
    
    return json.loads(response.choices[0].message.content)

def run_query(question):
    # Paso 1: Cargar datos
    index_data = load_index("data/embeddings_db.json")
    
    # Paso 2: Convertir pregunta a vector
    query_vector = get_embedding(question)
    
    # Paso 3: Búsqueda vectorial (k-NN)
    # Recuperamos los 3 mejores chunks
    relevant_chunks = vector_search(query_vector, index_data, top_k=3)
    
    # Paso 4: Generación de respuesta con LLM
    answer = generate_answer(question, relevant_chunks)

    # Llamamos al evaluador
    evaluation = evaluate_response(question, answer, [c["text"] for c in relevant_chunks])
    
    # Paso 5: Salida
    output = {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": [c["text"] for c in relevant_chunks],
        "evaluation": evaluation
    }
    
    return output

if __name__ == "__main__":
    # Prueba rápida
    user_input = input("Haz una pregunta sobre las políticas de HR: ")
    result = run_query(user_input)
    print(json.dumps(result, indent=4, ensure_ascii=False))