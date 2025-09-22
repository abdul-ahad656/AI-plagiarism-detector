AI-Based Plagiarism Checker

This project is a full-stack plagiarism detection system that uses advanced natural language processing techniques and machine learning models to assess the originality of textual content. It supports multiple use cases including direct document comparison, similarity against a reference corpus, and AI-generated content detection.
It is designed for researchers, educators, developers, and content reviewers seeking a fast, intuitive, and locally hosted plagiarism checker.


->Overview:

The system includes:
Document-to-document similarity scoring using BERT-based sentence embeddings
Corpus-based comparison for identifying overlap with existing saved documents
AI content detection using a Support Vector Machine classifier trained on human vs. GPT-2 generated samples
A React-based web interface styled with Tailwind CSS
A Python Flask backend that performs the core logic and model inference


->Project Structure:

The application is structured as follows:

backend/
├── app.py – Main Flask application
├── src/
│ ├── pipeline.py – Handles input cleaning and similarity computations
│ └── models/
│ ├── bert_trainer.py – Script for fine-tuning BERT (optional)
│ ├── similarity.py – Cosine similarity calculations
│ └── embedding_generator.py – Corpus embedding script
├── data/
│ ├── corpus/ – Saved user documents for comparison
│ └── raw/ – Placeholder for training or seed data
├── models/
│ ├── embeddings/ – Pickled BERT embeddings for the corpus
│ └── svm/ – Trained SVM model for AI detection

frontend/
├── public/
├── src/
│ ├── components/
│ │ ├── UploadForm.jsx
│ │ ├── SmartCheckForm.jsx
│ │ └── AIContentCheckForm.jsx
│ └── App.js – Client-side routes and page structure


->Features:

Compare two documents and identify semantic similarity
Upload a document and check against an internal repository (corpus)
Automatically save new original documents to the corpus
Detect the likelihood of AI-generated content in an uploaded file
Upload support for .txt, .pdf, and .docx file formats
Fully offline operation – no third-party APIs required
Modern, clean web interface with real-time visual feedback


->Setup Instructions:

Set up the backend:
**
cd backend
python -m venv venv
venv\Scripts\activate   # For Windows
pip install -r requirements.txt
**

Set up the frontend:
**
cd frontend
npm install
**

Run the backend:
**
cd backend
python app.py
**

Run the frontend:
**
cd frontend
npm run dev
**


->Environment Variables:

Create a .env file inside the frontend directory:
REACT_APP_API_URL=http://localhost:5000
This URL points to your locally running Flask backend.


->AI Detection Model:

The system uses a binary SVM model trained on labeled text samples from human authors and GPT-based language models. It evaluates the probability that a given file was written by AI. You can retrain the model by replacing the dataset and updating the model inside:
backend/models/svm/svm_ai_detector.pkl


->Visual Interface:

The user interface includes:
A homepage with navigation
Pages for document upload, AI detection, and corpus management
Donut-style similarity visualizations using React charts
Animated loading states for smoother UX


->Future Enhancements:

Admin panel for managing the corpus
Document chunking for better results on long texts
Downloadable PDF report of comparison results
Authentication and access control