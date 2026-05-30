# HR SaaS - Chatbot RAG de FAQ

Este proyecto implementa un sistema de Generación Aumentada por Recuperación (RAG) para responder preguntas frecuentes de Recursos Humanos.

## Estructura
- `data/`: Contiene el documento fuente y la base de datos vectorial (JSON).
- `src/`: Scripts de indexación y consulta.
- `outputs/`: Ejemplos de salida del sistema.

## Decisiones Técnicas
- **Chunking**: Se utilizó `RecursiveCharacterTextSplitter` con un tamaño de 500 y solapamiento de 100. Se eligió este método porque preserva mejor la unidad de las oraciones en comparación con un corte fijo.
- **Búsqueda Vectorial**: Se implementó k-Nearest Neighbors (k-NN) con Similitud Coseno utilizando Scikit-learn. Es ideal para este volumen de datos por su precisión exacta.

## Instalación y Uso
1. Instalar dependencias: `pip install -r requirements.txt`
2. Configurar `.env` con su `OPENAI_API_KEY`.
3. Indexar: `python src/build_index.py`
4. Consultar: `python src/query.py`