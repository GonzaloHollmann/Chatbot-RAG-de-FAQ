# Chatbot de Soporte FAQ - HR SaaS (RAG System)

Este proyecto consiste en la implementación de un sistema de **Generación Aumentada por Recuperación (RAG)** diseñado para automatizar las respuestas a preguntas frecuentes (FAQs) en una empresa de HR SaaS. El sistema procesa documentación no estructurada, la segmenta de forma inteligente y utiliza búsqueda vectorial para recuperar información precisa, permitiendo que un LLM genere respuestas fundamentadas (grounded) sin alucinaciones.

## 🚀 Instalación y Configuración

### Requisitos previos
- **Python 3.9 o superior**
- Una clave de API de OpenAI (`OPENAI_API_KEY`).

### Pasos para el Setup

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/GonzaloHollmann/Chatbot-RAG-de-FAQ.git
   cd chatbot-rag-faq
   ```

2. **Crear y activar entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto (puedes guiarte con `.env.example`):
   ```text
   OPENAI_API_KEY=tu_clave_aqui
   EMBEDDING_MODEL=text-embedding-3-small
   ```

## 🛠️ Uso del Sistema

El sistema funciona en dos etapas consecutivas:

1. **Pipeline de Indexación**
   Procesa el documento de texto, genera los fragmentos (chunks) y sus respectivos embeddings, guardándolos en una base de datos local.
   ```bash
   python src/build_index.py
   ```
   Este comando genera el archivo `data/embeddings_db.json` con más de 20 chunks procesados.

2. **Pipeline de Consulta (Chatbot)**
   Inicia la interfaz de consola para realizar preguntas al chatbot.
   ```bash
   python src/query.py
   ```

**Ejemplo de flujo de salida (JSON):**
```json
{
    "user_question": "¿Cómo puedo restablecer mi contraseña?",
    "system_answer": "Para restablecer tu contraseña, haz clic en '¿Olvidó su contraseña?' en la pantalla de inicio...",
    "chunks_related": [
        "Si no recuerdas tu contraseña, ve a la pantalla de inicio de sesión de BizFlow OS y haz clic en '¿Olvidó su contraseña?'..."
    ],
    "evaluation": {
        "score": 10,
        "reason": "La respuesta es precisa, utiliza los chunks proporcionados y responde directamente la duda del usuario."
    }
}
```

## 📂 Estructura del Proyecto

- `data/faq_document.txt`: Documento fuente con políticas y procedimientos (+1000 palabras).
- `data/embeddings_db.json`: Almacenamiento de los chunks y sus representaciones vectoriales.
- `src/build_index.py`: Lógica de carga, limpieza, chunking y generación de embeddings.
- `src/query.py`: Lógica de búsqueda vectorial, orquestación del LLM y agente evaluador.
- `outputs/sample_queries.json`: Archivo con 3 ejemplos de prueba para auditoría.
- `.env.example`: Plantilla de configuración de variables de entorno.
- `requirements.txt`: Lista de dependencias y versiones.

## 🧠 Decisiones Técnicas

- **Estrategia de Chunking:** Se implementó `RecursiveCharacterTextSplitter` con un `chunk_size` de 500 caracteres y un `chunk_overlap` de 100. Se eligió este método recursivo porque intenta mantener la integridad de párrafos y oraciones, evitando cortes que rompan el sentido semántico. El solapamiento (overlap) garantiza que el contexto compartido entre fragmentos no se pierda.
- **Búsqueda Vectorial:** Se utilizó k-Nearest Neighbors (k-NN) mediante Similitud Coseno (vía Scikit-learn). Dado el volumen de datos de una FAQ, este método proporciona una precisión exacta en la recuperación de los top-k fragmentos más relevantes.
- **Arquitectura RAG:** Se optó por RAG para resolver el problema de conocimiento limitado de los LLM. Esto permite actualizar la base de conocimientos simplemente modificando el archivo de texto, sin necesidad de reentrenar el modelo, asegurando respuestas veraces basadas en documentos oficiales.
- **Agente Evaluador:** Se incluyó un evaluador automático que puntúa la respuesta (0-10) analizando la fidelidad de la respuesta frente a los chunks recuperados, aportando una capa extra de aseguramiento de calidad.
