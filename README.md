# EmoSense: Text Emotion Analysis

A machine learning application that analyzes emotions in text by predicting **Valence** (positivity/negativity) and **Arousal** (intensity/calmness) on a 1-5 scale.

## Project Overview

EmoSense uses natural language processing and regression models to understand the emotional tone of any given text. It provides a user-friendly web interface built with Streamlit where users can input text and receive detailed emotion analysis with visual feedback.

### Dimensions of Emotion

- **Valence (1-5 scale)**: Measures positivity vs. negativity
  - 1.0-2.5: Negative/Unpleasant
  - 2.5-3.5: Neutral
  - 3.5-5.0: Positive/Pleasant

- **Arousal (1-5 scale)**: Measures emotional intensity/energy
  - 1.0-2.5: Low/Calm
  - 2.5-3.5: Moderate
  - 3.5-5.0: High/Excited

## Project Structure

```
sentiment analysis/
├── Code/
│   ├── app.py                      # Streamlit web interface (current)
│   └── trainmodel.py               # Model training script (current)
├── dataset/
│   └── emobank.csv                 # EmoBank dataset
├── Xgboost_legacy/                 # Previous implementation with XGBoost
│   ├── app.py                      # Streamlit web interface (legacy)
│   ├── trainmodel.py               # Model training script (legacy)
│   ├── emotion_model.pkl           # Trained XGBoost model (legacy)
│   └── tfidf_vectorizer.pkl        # Text vectorizer (legacy)
├── emotion_model.pkl               # Trained Ridge Regression model (current)
├── tfidf_vectorizer.pkl            # Text vectorizer (current)
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application for emotion analysis |
| `trainmodel.py` | Trains the Ridge Regression model on EmoBank dataset |
| `emobank.csv` | Dataset with text samples and emotion labels (V, A) |
| `emotion_model.pkl` | Serialized trained model (generated after training) |
| `tfidf_vectorizer.pkl` | Serialized text vectorizer (generated after training) |
| `requirements.txt` | Lists all Python package dependencies |
| `README.md` | Complete project documentation |
| `Xgboost_legacy/` | Alternative XGBoost implementation (optional) |

## Current Implementation

### Model: Ridge Regression
- **Algorithm**: Ridge Regression with MultiOutputRegressor
- **Vectorizer**: TfidfVectorizer with uni-grams and bi-grams
- **Outputs**: Two simultaneous predictions (Valence and Arousal)
- **Training/Test Split**: 80/20

### Key Files

#### `trainmodel.py`
Trains the emotion prediction model:
1. Loads the EmoBank dataset from CSV
2. Cleans data (removes missing text entries)
3. Vectorizes text using TF-IDF (with n-grams for better context)
4. Splits data into training and testing sets
5. Trains Ridge regression model with MultiOutputRegressor
6. Evaluates Mean Absolute Error (MAE) for both dimensions
7. Saves trained model and vectorizer as pickle files

**Output Files**:
- `emotion_model.pkl` - Trained regression model
- `tfidf_vectorizer.pkl` - Text vectorizer for inference

#### `app.py`
Interactive Streamlit web application:
- Loads pre-trained model and vectorizer
- Accepts user text input
- Predicts emotion scores
- Displays results with:
  - Numerical scores (0-2 decimal places)
  - Progress bars for visual representation
  - Emotion interpretation (Positive/Negative/Neutral, High/Moderate/Low energy)
- Score scaling ensures full utilization of 1-5 range

## Legacy Version

### Xgboost_legacy/
Previous implementation using XGBoost instead of Ridge Regression:
- Uses `xgb.XGBRegressor` with enhanced parameters (200 trees, depth 7)
- Implements emotion amplification (3.5x multiplier) for more pronounced scoring
- Captures diagnostic information (recognized keywords)

To use the legacy version, run scripts from the `Xgboost_legacy/` directory.

## Installation & Setup

### Requirements
- Python 3.7+
- All dependencies listed in `requirements.txt`

### Option 1: Using pip (Recommended)

1. **Create a virtual environment**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Option 2: Using conda

1. **Create a conda environment**:
```bash
conda create -n emosense python=3.10
conda activate emosense
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
# OR
conda install pandas scikit-learn joblib streamlit numpy xgboost
```

### Option 3: Manual Installation

```bash
pip install pandas>=1.5.0 scikit-learn>=1.2.0 joblib>=1.2.0 streamlit>=1.28.0 numpy>=1.24.0 xgboost>=2.0.0
```

## Quick Start

Get the app running in 3 simple steps:

### 1. Train the Model
```bash
python trainmodel.py
```
**Output**: Two files are created
- `emotion_model.pkl` - Trained emotion prediction model
- `tfidf_vectorizer.pkl` - Text vectorizer for processing input

**Console Output**:
```
Loading dataset...
Vectorizing text...
Training model...

Valence MAE: 0.XXX (Lower is better)
Arousal MAE: 0.XXX (Lower is better)

Models saved successfully.
```

### 2. Launch the Web App
```bash
streamlit run app.py
```
The app automatically opens at `http://localhost:8501`

### 3. Analyze Emotions
1. Type any text in the input box
2. Click "Predict Emotion 🚀"
3. See the results with scores and visual feedback

**Example**:
```
Input: "I'm absolutely thrilled about this amazing opportunity!"

Output:
Valence Score: 4.25 / 5.0 (Positive / Pleasant)
Arousal Score: 4.10 / 5.0 (High / Excited)
```

## Configuration

### Model Parameters (in `trainmodel.py`)

#### Ridge Regression Settings
```python
base_model = Ridge(alpha=1.0)  # Regularization strength
```
- **alpha**: Controls model complexity (higher = stronger regularization)
- Adjust if model overfits or underfits

#### TF-IDF Vectorizer Settings
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
```
- **ngram_range=(1, 2)**: Uses single words and 2-word phrases
- Change to `(1, 3)` to include 3-word phrases (slower, more features)
- Change to `(1, 1)` for single words only (faster, fewer features)

#### Train/Test Split
```python
train_test_split(..., test_size=0.2, random_state=42)
```
- **test_size=0.2**: 20% of data for testing, 80% for training
- **random_state=42**: Ensures reproducible results

### Streamlit App Configuration (in `app.py`)

#### Score Scaling Parameters
```python
def scale_score(raw_score, raw_min=2.4, raw_max=3.6, target_min=1.0, target_max=5.0):
```
- **raw_min/raw_max**: Expected bounds of model output
- **target_min/target_max**: Desired display range (1.0 to 5.0)
- Adjust raw bounds if scores consistently exceed 1-5 range

#### Emotion Interpretation Thresholds
```python
if valence >= 3.5:
    st.write("🟢 **Vibe: Positive / Pleasant**")
```
- **3.5**: Threshold for positive classification
- Adjust thresholds to fine-tune emotion categories

### Legacy (XGBoost) Configuration (in `Xgboost_legacy/trainmodel.py`)

```python
base_model = xgb.XGBRegressor(
    n_estimators=200,      # Number of boosting rounds
    max_depth=7,           # Tree depth (higher = more complex)
    learning_rate=0.1,     # Step size for boosting
    objective='reg:squarederror'
)
```

#### Emotion Amplification (XGBoost only)
```python
AMPLIFIER = 3.5
amplified_valence = 3.0 + ((raw_valence - 3.0) * AMPLIFIER)
```
- **AMPLIFIER**: Stretches scores away from neutral (3.0)
- Higher values = more extreme predictions
- Set to 1.0 to disable amplification

## Usage

For detailed step-by-step instructions, see **Quick Start** section above.

### Command Reference

| Task | Command |
|------|---------|
| Train model | `python trainmodel.py` |
| Launch app | `streamlit run app.py` |
| Use legacy version | `cd Xgboost_legacy && python trainmodel.py` |
| Stop Streamlit | `Ctrl+C` |
| Deactivate environment | `deactivate` (or `conda deactivate`) |

## Dataset

**EmoBank CSV Format**:
```csv
text,V,A
"Sample text here",3.2,2.8
...
```

- `text`: Raw text to analyze
- `V`: Valence score (ground truth, 1-5 range)
- `A`: Arousal score (ground truth, 1-5 range)

## Model Performance

The current Ridge model outputs MAE (Mean Absolute Error) metrics:
- Lower MAE values indicate better accuracy
- Both Valence and Arousal are evaluated independently

Example output:
```
Valence MAE: 0.543
Arousal MAE: 0.621
```

## Technical Details

### Text Processing Pipeline
1. **TF-IDF Vectorization**: Converts text to numerical features
2. **N-grams**: Uses 1-grams and 2-grams to capture word context
   - Example: "really sad" is recognized as a meaningful phrase
3. **Sparse Matrix**: Efficient representation of text features

### Prediction Pipeline
1. User input text is vectorized using the trained vectorizer
2. Model predicts raw scores for Valence and Arousal
3. Scores are scaled to 1-5 range for consistency
4. Results displayed with visual feedback

## Differences: Current vs. Legacy

| Feature | Current (Ridge) | Legacy (XGBoost) |
|---------|-----------------|-----------------|
| Algorithm | Ridge Regression | XGBoost |
| Training Speed | Fast | Slower (200 trees) |
| Complexity | Simpler | More complex |
| Score Amplification | Linear scaling | 3.5x amplifier |
| Diagnostic Info | Basic | Includes keyword count |

## Future Improvements

- Add sentiment categories (e.g., anger, joy, sadness)
- Implement attention mechanisms for explainability
- Support for multiple languages
- Real-time model updates
- API endpoints for external integration
- Model versioning and comparison

## Notes

- Ensure dataset file path is correct before training
- Model files must be in the same directory as `app.py`
- For legacy version, adjust file paths accordingly (use `../dataset/`)
- Text vectorizer is dataset-specific; retrain if using different data

## License

This project uses the EmoBank dataset. Refer to the dataset documentation for usage rights and citations.
