# ⚽ FIFA World Cup 2026 Predictor

A machine learning-powered application that predicts international football (soccer) match outcomes and simulates World Cup tournaments using XGBoost classification and Monte Carlo simulations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **⚽ Match Predictor** - Predict head-to-head match outcomes with probability distributions
- **🏆 Tournament Simulator** - Run Monte Carlo simulations (100-5000 iterations) to estimate World Cup winning odds
- **📊 Team Analytics** - Detailed team statistics including ELO ratings, form scores, and performance metrics
- **📈 Team Rankings** - Browse global rankings sorted by ELO, form, goals scored, or defensive strength
- **🎨 Modern UI** - Professional dark-mode interface with gradient design and interactive visualizations
- **🔮 Confidence Levels** - AI-powered confidence indicators for prediction reliability
- **📊 Head-to-Head Analysis** - Compare team statistics with detailed performance breakdowns

## 🛠️ Technology Stack

### Core ML & Data Processing
- **XGBoost** - Multi-class classification model for match outcome prediction
- **Pandas** - Data manipulation and team statistics aggregation
- **NumPy** - Numerical computations
- **Scikit-learn** - Model evaluation and train-test splitting

### Frontend & Visualization
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive charts and visualizations
- **Matplotlib & Seaborn** - Static visualizations

### Other
- **Joblib** - Model serialization and loading

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/FIFA2026-Predictor.git
cd FIFA2026-Predictor
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Command Line Prediction

Use the script to predict a match from the terminal:

```bash
python src/predict_match.py
```

Follow the prompts to enter home and away team names.

## 📊 Project Structure

```
FIFA2026-Predictor/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/                          # Source code modules
│   ├── train_xgboost.py         # Model training script
│   ├── feature_engineering.py   # Feature extraction pipeline
│   ├── enhance_features.py      # Feature enhancement
│   ├── create_team_stats_v2.py  # Team statistics aggregation
│   ├── predict_match.py         # Single match prediction
│   ├── simulate_match.py        # Individual match simulation
│   └── simulate_worldcup.py     # Tournament simulation
│
├── data/                         # Datasets
│   ├── results.csv              # Historical match results
│   ├── elo_ratings.csv          # ELO ratings over time
│   ├── engineered_matches.csv   # Processed features
│   ├── enhanced_matches.csv     # Enhanced feature set
│   ├── team_stats.csv           # Aggregated team statistics
│   └── worldcup_teams.txt       # Participating nations
│
├── models/                       # Trained models
│   └── worldcup_xgb.pkl         # XGBoost classifier (trained)
│
├── notebooks/                    # Jupyter notebooks
│   └── [exploration notebooks]
│
└── archive/                      # Previous versions
    ├── train.py
    ├── train_xgboost_v2.py
    └── debug.py
```

## 🧠 How It Works

### 1. **Data Pipeline**
- Loads historical match results and ELO ratings
- Aggregates team statistics from match history
- Extracts features like form, goals scored, goals conceded
- Normalizes and engineers features for ML model

### 2. **Feature Engineering** (9 Features)
```
- Home Team Form Score
- Away Team Form Score
- Home Goals Scored (Last 5 Matches)
- Away Goals Scored (Last 5 Matches)
- Home Goals Conceded (Last 5 Matches)
- Away Goals Conceded (Last 5 Matches)
- Home Team ELO Rating
- Away Team ELO Rating
- ELO Difference (Home - Away)
```

### 3. **Model Architecture**
- **Algorithm**: XGBoost Classifier
- **Task**: Multi-class classification (3 outcomes)
- **Classes**: 
  - 0: Away Win
  - 1: Draw
  - 2: Home Win
- **Configuration**:
  - Estimators: 500
  - Max Depth: 6
  - Learning Rate: 0.05
  - Objective: Multi-class softmax probability

### 4. **Prediction Process**
1. Select two teams from database
2. Extract team statistics
3. Create feature vector
4. Feed to XGBoost model
5. Get probability distribution (0-100%)
6. Visualize results with confidence level

### 5. **Tournament Simulation**
- Runs multiple simulations (configurable: 100-5000)
- Each simulation:
  - Randomly shuffles team order
  - Simulates matches using probabilities
  - Progresses teams to next round
  - Determines champion
- Aggregates results to show winning probabilities

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~58-62% |
| Training Samples | 500+ historical matches |
| Features Used | 9 statistical metrics |
| Model Type | XGBoost (Multi-class) |
| Cross-validation | 5-fold stratified |

## 🎯 Application Pages

### 🏠 Dashboard
- Quick statistics overview
- Top 8 teams by ELO rating
- Quick navigation buttons
- Model information

### ⚽ Match Predictor
- Select any two teams
- View detailed team comparison
- Get match outcome probabilities
- See confidence levels
- Analyze performance metrics

### 🏆 Tournament Simulator
- Run Monte Carlo simulations
- Configure number of iterations
- Select tournament participants
- View championship odds
- Analyze probability distribution

### 📈 Team Rankings
- Browse all teams sorted by metrics
- Filter by ELO, form, goals scored, or defense
- Top 15 visualization
- Offensive vs defensive scatter plot

### 📊 Analytics
- Deep dive into individual team statistics
- Performance gauges and metrics
- Head-to-head predictions vs multiple opponents
- Detailed statistical breakdown

## 💡 Usage Examples

### Example 1: Predict Argentina vs Brazil
```
Home Team: Argentina
Away Team: Brazil
```
**Output:**
- Argentina Win: 45.67%
- Draw: 28.34%
- Brazil Win: 25.99%
- Confidence: High

### Example 2: Run Tournament Simulation
```
Teams: [Argentina, Brazil, France, Spain, England, Portugal, Germany, Netherlands]
Simulations: 1000
```
**Output:**
```
Champion Odds:
1. Argentina: 28.5%
2. France: 24.3%
3. Brazil: 22.1%
...
```

## 🔧 Configuration & Customization

### Add New Teams
1. Update `data/worldcup_teams.txt`
2. Add team statistics to `data/team_stats.csv`
3. Retrain model or manually input stats

### Adjust Model Hyperparameters
Edit `src/train_xgboost.py`:
```python
model = XGBClassifier(
    n_estimators=500,      # Number of trees
    max_depth=6,           # Tree depth
    learning_rate=0.05,    # Learning rate
    # ... other parameters
)
```

### Customize UI Theme
Edit the CSS section in `streamlit_app.py` (lines 38-150)

## 📈 Data Sources

- **Historical Matches**: FIFA/International Football Results Database
- **ELO Ratings**: World ELO Ratings (https://eloratings.net/)
- **Team Statistics**: Aggregated from match data

## 🚀 Future Improvements

- [ ] Add recent 2026 World Cup qualification data
- [ ] Implement ensemble methods (combine multiple models)
- [ ] Add player-level statistics
- [ ] Real-time data updates from API
- [ ] Mobile app version
- [ ] Betting odds integration
- [ ] Advanced analytics dashboard
- [ ] Historical prediction accuracy tracking
- [ ] Team form decay factor
- [ ] Home/Away advantage modeling

## 🐛 Troubleshooting

### Issue: "Model file not found"
**Solution**: Ensure `models/worldcup_xgb.pkl` exists. Run `src/train_xgboost.py` to retrain.

### Issue: "Team not found"
**Solution**: Check `data/team_stats.csv` for correct team name spelling.

### Issue: Streamlit not starting
**Solution**: 
```bash
pip install --upgrade streamlit
streamlit run streamlit_app.py --logger.level=debug
```

### Issue: Poor prediction accuracy
**Solution**: Data may be outdated. Retrain with recent match data using `src/train_xgboost.py`

## 📝 Requirements

See [requirements.txt](requirements.txt) for detailed version information.

Key dependencies:
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=0.24.0
- xgboost>=1.5.0
- streamlit>=1.0.0
- plotly>=5.0.0
- joblib>=1.0.0

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Data collection and validation
- Model improvements and tuning
- UI/UX enhancements
- Documentation improvements
- Bug fixes and optimizations
- New visualization types

## 📧 Contact & Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for Q&A and ideas

## 🙏 Acknowledgments

- XGBoost team for the powerful gradient boosting framework
- Streamlit team for the amazing web framework
- Plotly for interactive visualizations
- FIFA and international football community for historical data

## ⭐ Show Your Support

If you found this project helpful, please consider:
- Giving it a star ⭐
- Sharing it with others
- Contributing improvements
- Using it for your research/analysis

---

**Made with ❤️ for football fans and ML enthusiasts**

Last Updated: June 2026
# FIFA2026-Predictor
