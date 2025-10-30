title: Plataforma Citas Demo emoji: 🗓️ sdk: gradio app_file: app.py pinned: false

🤖 Plataforma de Citas por Voz y Chat

Este prototipo fue desarrollado para el curso de Servicios Cognitivos in Cloud. Implementa un asistente virtual con procesamiento de lenguaje natural (NLP) para gestionar citas médicas.

Características Principales (Sprint 3)

Interfaz Conversacional: Utiliza Gradio para permitir la interacción vía chat y voz.

Lógica de Negocio (CRUD): Las funciones de Crear, Consultar y Cancelar citas operan en tiempo real, conectándose a una base de datos externa (Google Sheets) mediante un sistema seguro de "Secrets" (variables de entorno).

Procesamiento de Lenguaje Natural (NLP): Usa spaCy para detectar la intención del usuario (Agendar, Consultar, Cancelar) y extraer datos clave (DNI, Fecha, Médico).

Predicción de No-Show (ML): Incluye un modelo de Machine Learning (Regresión Logística) que evalúa el riesgo de ausencia de cada nueva cita y emite una advertencia.

Integración de Voz (STT): Permite la transcripción de audio mediante faster-whisper.

Ejecución Local

Para correr la aplicación en su máquina:

Clonar y configurar:

# Clonar el repositorio
git clone [https://www.youtube.com/watch?v=PDRLx0cYzbM](https://www.youtube.com/watch?v=PDRLx0cYzbM)
cd plataforma-citas

# Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate


Instalar dependencias:

pip install -r requirements.txt

# Descargar modelo base de spaCy (necesario para extracción de entidades)
python -m spacy download es_core_news_sm


Ejecutar la App:

python app.py


(La aplicación se abrirá en http://127.0.0.1:7860).

Nota: El código en flujo_agendamiento.py y entrenar_nlp.py debe estar modificado para leer las credenciales del "Secret" GOOGLE_CREDENTIALS_JSON  cuando se ejecuta en Hugging Face.