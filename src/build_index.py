import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# 1. Cargar variables de entorno (API Key)
load_dotenv()
client = OpenAI()

def load_document(file_path):
    """Lee el contenido del archivo de texto."""
    print(f"Cargando documento desde: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_chunks(text):
    """Divide el texto en trozos pequeños con solapamiento."""
    # Configuramos el splitter
    # chunk_size: tamaño del trozo (en caracteres)
    # chunk_overlap: cuánto se repite del trozo anterior para no perder contexto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Documento dividido en {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks):
    """Convierte cada trozo de texto en un vector numérico usando OpenAI."""
    print("Generando embeddings...")
    embeddings_data = []
    
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        # Guardamos el texto original junto con su vector
        embeddings_data.append({
            "id": i,
            "text": chunk,
            "embedding": response.data[0].embedding
        })
    return embeddings_data

def save_index(data, output_path):
    """Guarda los chunks y embeddings en un archivo JSON para usarlos luego."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Índice guardado exitosamente en: {output_path}")

def run_indexing():
    # Paso 1: Cargar
    raw_text = load_document("data/faq_document.txt")
    
    # Paso 2: Chunking
    chunks = create_chunks(raw_text)
    
    # Validación de la consigna: mínimo 20 chunks
    if len(chunks) < 20:
        print("⚠️ Advertencia: Tienes menos de 20 chunks. Considera reducir el 'chunk_size'.")
    
    # Paso 3: Embeddings
    data_with_vectors = generate_embeddings(chunks)
    
    # Paso 4: Almacenar
    # Lo guardamos en 'data/' para que el otro script lo encuentre
    save_index(data_with_vectors, "data/embeddings_db.json")

if __name__ == "__main__":
    run_indexing()