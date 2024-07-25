# **French Practicing Assistant: RAG-Powered French Learning Chatbot**

French Practicing Assistant is an advanced chatbot designed to facilitate French language learning through an interactive and engaging interface. Powered by the Zephyr 7B Beta language model and enhanced with Retrieval-Augmented Generation (RAG), this application offers a unique, personalized French learning experience based on the book "Easy French Step-by-Step" by Myrna Bell Rochester.

### Key Features:
- **Intelligent French Instruction**: The chatbot functions as a dedicated French tutor, offering:
  - Explanations of French grammar concepts
  - Vocabulary assistance
  - Pronunciation guidance
  - Practice exercises
  - Simple French conversations for practice
- **RAG-Enhanced Responses**: Utilizes a vector database built from "Easy French Step-by-Step" to provide contextually relevant information
- **Bilingual Communication**: Adapts to the user's level, communicating in both English and French
- **Customizable Learning Experience**: The chatbot's behavior can be fine-tuned through the system message

### Technical Details:
- Built with Python using the Gradio library for the user interface
- Utilizes Hugging Face's Inference API to interact with the Zephyr 7B Beta model
- Implements RAG using:
  - PyMuPDF for PDF processing
  - Sentence Transformers for text embedding
  - FAISS for efficient similarity search
- Allows adjustment of model parameters such as max tokens, temperature, and top-p sampling

### Usage:
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the application**:
   ```bash
   python app.py
   ```
3. **Access the French Practicing Assistant interface** via the provided URL
4. **Begin your French learning journey** by engaging with the chatbot

### Key Components:
- **PDF Processing**: Extracts text from "Easy French Step-by-Step" PDF
- **Vector Database**: Creates embeddings for efficient information retrieval
- **RAG Integration**: Enhances responses with relevant information from the book

### Customization:
The chatbot's core functionality is defined in the system message within the `respond` function. This can be modified to adjust the chatbot's behavior and focus areas.

### Note:
French Practicing Assistant is an educational project. While it offers a robust platform for French language learning based on "Easy French Step-by-Step", it is not a substitute for a human French teacher or immersive language learning.

### Disclaimer:
This chatbot is based on 'Easy French Step-by-Step' by Myrna Bell Rochester. It's for educational purposes only and not a substitute for a human French teacher or immersive language learning.

For any questions or issues, please open an issue in the GitHub repository.
