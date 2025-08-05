Project Overview:
  An NLP-powered plant disease diagnosis tool that predicts diseases based on textual symptom descriptions (e.g., "yellow leaves with brown spots"). 
  Built with:
  Scikit-learn's MultinomialNB (Naive Bayes) for classification
  Streamlit for a simple web interface

Key Features:
  Symptom-to-Disease Prediction: Input text → Get probable disease
  Multi-Crop Support: Works for tomatoes, potatoes, wheat, etc.
  Lightweight & Fast: Naive Bayes ensures quick predictions
  User-Friendly UI: No ML knowledge needed 
Prerequisites:
 pip install streamlit pyngrok --quiet 
 pip install -q pyngrok streamlit
 
App ScreenShot:
<img width="1798" height="861" alt="image" src="https://github.com/user-attachments/assets/2f178274-4268-4d4e-9cc1-a512262a746c" />

Code Structure:
plant_disease_detector/  
├── data/                    # CSV/JSON datasets (symptom-disease pairs)  
├── model/                   # Saved MultinomialNB model (.pkl file)  
├── app.py                   # Streamlit application  
├── train.py                 # Model training script  
└── requirements.txt         # Libraries

How It Works:-

Text Preprocessing:
Tokenization, stopword removal, TF-IDF vectorization
Model Prediction:
MultinomialNB compares input symptoms with trained patterns
Output:
Returns top disease matches
