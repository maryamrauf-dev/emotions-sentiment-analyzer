import streamlit as st
import pandas as pd
import io
# Import the logic from our separate file
from emotion_logic import load_classifier, preprocess_text, analyze_single_text

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Emotion AI Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700&family=Orbitron:wght@400;700&display=swap');

    :root {
      --main-bg: #21112D;
      --accent-pink: #FF007F;
      --glow-purple: #7F00FF;
      --light-text: #FFFFFF;
      --glass-bg: rgba(33, 17, 45, 0.7);
    }

    .stApp {
        background-color: var(--main-bg);
        color: var(--light-text);
        font-family: 'Outfit', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: rgba(26, 10, 35, 0.95);
        border-right: 2px solid var(--glow-purple);
    }

    h1, h2, h3, .neon-text {
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px var(--accent-pink), 0 0 20px var(--accent-pink);
        color: var(--accent-pink) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .neon-title {
        font-size: 3rem !important;
        margin-bottom: 2rem !important;
        text-align: center;
    }

    .glass-panel {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(127, 0, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .history-card {
        padding: 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }

    .history-text {
        font-size: 0.8rem;
        font-style: italic;
        margin-bottom: 8px;
        color: white;
    }

    .history-badge {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.65rem;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        background: var(--accent-pink);
        color: white;
    }

    .stButton > button, 
    [data-testid="stFileUploader"] button {
        background: linear-gradient(45deg, var(--accent-pink), var(--glow-purple)) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button {
        width: 100%;
        padding: 18px 40px !important;
    }

    .stButton > button:hover,
    [data-testid="stFileUploader"] button:hover {
        box-shadow: 0 0 30px var(--accent-pink) !important;
        transform: translateY(-2px) !important;
    }

    /* Style Text Area and File Uploader to match sidebar */
    textarea, [data-testid="stFileUploader"] > div {
        background-color: rgba(33, 17, 45, 0.7) !important;
        border: 1px solid var(--accent-pink) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 15px !important;
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.2) !important;
    }

    textarea:focus {
        border-color: var(--glow-purple) !important;
        box-shadow: 0 0 20px var(--accent-pink) !important;
        transform: scale(1.005);
        transition: all 0.3s ease;
    }

    /* Style the file uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(127, 0, 255, 0.5) !important;
        background: transparent !important;
    }

    .emotion-badge {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        text-transform: uppercase;
        color: var(--accent-pink);
        text-shadow: 0 0 10px var(--accent-pink), 0 0 20px var(--accent-pink);
        text-align: center;
        display: block;
        padding: 20px 0;
    }

    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, var(--accent-pink), var(--glow-purple)) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def get_classifier():
    return load_classifier()

classifier = get_classifier()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">'
                '<span style="font-size: 2rem;">🧠</span>'
                '<h1 style="font-size: 1.2rem; margin: 0;">NEURAL CORE</h1>'
                '</div>', unsafe_allow_html=True)
    
    st.markdown("### MODEL SPECIFICATIONS")
    st.markdown("""
    <div class="glass-panel" style="padding: 15px; border: 1px solid #FF007F;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #ccc; font-size: 0.7rem;">MODEL:</span>
            <span style="color: #FF007F; font-weight: bold; font-size: 0.75rem;">XLNet-Base-Cased</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #ccc; font-size: 0.7rem;">ARCHITECTURE:</span>
            <span style="color: #FF007F; font-weight: bold; font-size: 0.75rem;">Transformer-XL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### RECENT HISTORY")
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if not st.session_state.history:
        st.write("No recent analysis")
    else:
        for item in st.session_state.history:
            st.markdown(f"""
            <div class="history-card">
                <p class="history-text">"{item['text'][:40]}..."</p>
                <span class="history-badge">{item['emotion'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown('<h1 class="neon-text neon-title">EMOTION AI ANALYZER</h1>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["SINGLE ANALYSIS", "BATCH PROCESSING"])

with tab1:
    user_input = st.text_area("ENTER TEXT TO ANALYZE...", height=200, key="single_input")
    
    if st.button("DETECT EMOTION", key="btn_single"):
        if user_input.strip():
            with st.spinner("ANALYZING..."):
                results = analyze_single_text(classifier, user_input)
                top_emotion = results[0]['label']
                
                # Add to history
                st.session_state.history = ([{"text": user_input, "emotion": top_emotion}] + st.session_state.history)[:5]
                
                # Display results
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown(f'<span style="font-size: 0.9rem; letter-spacing: 3px; opacity: 0.7; text-align: center; display: block;">PRIMARY EMOTION DETECTED</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="emotion-badge">{top_emotion}</span>', unsafe_allow_html=True)
                
                st.markdown("### CONFIDENCE LEVELS")
                for pred in results:
                    st.write(f"{pred['label'].upper()} ({(pred['score'] * 100):.1f}%)")
                    st.progress(pred['score'])
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text first!")

with tab2:
    uploaded_file = st.file_uploader("DRAG & DROP CSV FILE", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find text column
            text_col = None
            for col in ['text', 'content', 'Tweet', 'sentence']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col:
                if st.button("PROCESS BATCH", key="btn_batch"):
                    with st.spinner("PROCESSING BATCH..."):
                        results_labels = []
                        for text in df[text_col]:
                            clean = preprocess_text(str(text))
                            pred = classifier(clean, top_k=1)[0]
                            results_labels.append(pred['label'])
                        
                        df['predicted_emotion'] = results_labels
                        summary = df['predicted_emotion'].value_counts()
                        
                        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                        st.markdown(f'<span style="font-size: 0.9rem; letter-spacing: 3px; opacity: 0.7; text-align: center; display: block;">BATCH ANALYSIS COMPLETE</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="emotion-badge">{len(df)} RECORDS</span>', unsafe_allow_html=True)
                        
                        st.markdown("### EMOTION DISTRIBUTION")
                        for label, count in summary.items():
                            percentage = count / len(df)
                            st.write(f"{label.upper()}: {count} items ({percentage:.1%})")
                            st.progress(percentage)
                        
                        st.markdown("### PREVIEW (TOP 10)")
                        st.dataframe(df[[text_col, 'predicted_emotion']].head(10), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("CSV must contain a 'text', 'content', or 'sentence' column.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
