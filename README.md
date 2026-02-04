# 🧠 Emotion AI Analyzer

A premium, cyberpunk-themed web application for detecting emotions in text using state-of-the-art AI. Built with **Streamlit** and **HuggingFace Transformers**.

![Design Preview](https://img.shields.io/badge/Design-Cyberpunk-FF007F)
![AI Model](https://img.shields.io/badge/Model-XLNet-7F00FF)

## ✨ Features
- **Neural Single Analysis**: Get instant sentiment feedback on any sentence.
- **Batch Processing**: Upload a CSV file to analyze hundreds of rows at once.
- **Cyberpunk UI**: A high-end dashboard with neon glows and glassmorphism.
- **Recent History**: Track your latest analysis directly in the sidebar.

## 📁 Project Structure
- `app.py`: The Main UI and layout code.
- `emotion_logic.py`: The Core AI logic and text cleaning.
- `requirements.txt`: List of necessary Python libraries.
- `run_app.bat`: One-click launcher for Windows.

## 🚀 How to Run

### Option 1: One-Click (Windows)
Simply double-click the `run_app.bat` file. It will automatically install missing libraries and start the app for you.

### Option 2: Manual Terminal
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Built with
- [Streamlit](https://streamlit.io/) - For the interactive UI.
- [Transformers](https://huggingface.co/docs/transformers/index) - Using XLNet/BERT for emotion classification.
- [Pandas](https://pandas.pydata.org/) - For CSV batch processing.

## 📝 Beginner Notes
- This project is designed to be **easy to read**.
- **No Complex Frameworks**: We avoid using FastAPI or Pydantic to keep things simple.
- **Modularity**: The UI and Logic are separated into two files so you can study how they talk to each other.
