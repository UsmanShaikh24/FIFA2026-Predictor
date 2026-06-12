import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import Counter
import numpy as np
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - PROFESSIONAL MODERN THEME
# =====================================================

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-bg: #0F1419;
        --card-bg: #1a1f2e;
        --accent-gold: #FFD700;
        --success-green: #00D084;
        --warning-orange: #FFA500;
        --danger-red: #FF6B6B;
    }

    body, .main {
        background: linear-gradient(135deg, #0F1419 0%, #1a1f2e 100%);
        color: #E0E0E0;
    }

    .main {
        padding: 0;
    }

    /* Headers */
    h1 {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3.5rem !important;
        letter-spacing: -1px;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        color: #FFD700 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #E0E0E0 !important;
        font-weight: 600 !important;
    }

    /* Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        padding: 1.5rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }

    div[data-testid="metric-container"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }

    /* Buttons */
    button {
        background: var(--primary-gradient) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        color: white !important;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }

    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5) !important;
    }

    /* Selectbox & Input */
    .stSelectbox, .stTextInput, .stNumberInput {
        border-radius: 12px !important;
    }

    /* Data frames */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1f2e 0%, #0F1419 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .main > div {
        animation: fadeIn 0.6s ease-out;
    }

    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        background: var(--primary-gradient);
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_team_flag(team_name):
    """Get flag emoji for teams"""
    flags = {
        "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷",
        "Spain": "🇪🇸", "England": "🇬🇧", "Portugal": "🇵🇹",
        "Germany": "🇩🇪", "Netherlands": "🇳🇱", "Italy": "🇮🇹",
        "Belgium": "🇧🇪", "Uruguay": "🇺🇾", "Croatia": "🇭🇷",
        "Poland": "🇵🇱", "Mexico": "🇲🇽", "Japan": "🇯🇵",
        "USA": "🇺🇸", "Canada": "🇨🇦", "Korea": "🇰🇷"
    }
    return flags.get(team_name, "⚽")

def predict_match(team1, team2, stats, model):
    """Predict match outcome"""
    home = stats.loc[team1]
    away = stats.loc[team2]
    
    X = [[
        home["form"],
        away["form"],
        home["goals_scored"],
        away["goals_scored"],
        home["goals_conceded"],
        away["goals_conceded"],
        home["elo"],
        away["elo"],
        home["elo"] - away["elo"]
    ]]
    
    return model.predict_proba(X)[0]

def get_confidence_level(max_prob):
    """Determine confidence level"""
    if max_prob >= 0.60:
        return "🔴 Very High Confidence"
    elif max_prob >= 0.50:
        return "🟠 High Confidence"
    elif max_prob >= 0.40:
        return "🟡 Moderate Confidence"
    else:
        return "🟢 Low Confidence"

def simulate_match(team1, team2, stats, model):
    """Simulate a single match"""
    probs = predict_match(team1, team2, stats, model)
    r = random.random()
    
    if r < probs[0]:
        return team2
    elif r < probs[0] + probs[1]:
        return random.choice([team1, team2])
    return team1

def simulate_tournament(teams, stats, model, num_sims=1000):
    """Simulate multiple tournaments"""
    champions = []
    
    for _ in range(num_sims):
        tournament_teams = teams.copy()
        random.shuffle(tournament_teams)
        
        # Quarterfinals
        semi = []
        for i in range(0, len(tournament_teams), 2):
            if i + 1 < len(tournament_teams):
                semi.append(simulate_match(tournament_teams[i], tournament_teams[i + 1], stats, model))
        
        # Semifinals
        final = []
        for i in range(0, len(semi), 2):
            if i + 1 < len(semi):
                final.append(simulate_match(semi[i], semi[i + 1], stats, model))
        
        # Final
        if len(final) == 2:
            champion = simulate_match(final[0], final[1], stats, model)
            champions.append(champion)
    
    return champions

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("models/worldcup_xgb.pkl")

@st.cache_data
def load_stats():
    return pd.read_csv("data/team_stats.csv", index_col=0)

model = load_model()
stats = load_stats()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("### 🌟 FIFA World Cup 2026")
st.sidebar.markdown("**Powered by Machine Learning**")
st.sidebar.divider()

page_options = [
    "🏠 Dashboard",
    "⚽ Match Predictor",
    "🏆 Tournament Simulator",
    "📈 Team Rankings",
    "📊 Analytics"
]

current_index = page_options.index(st.session_state.current_page)

selected_page = st.sidebar.radio(
    "📊 Navigation",
    page_options,
    index=current_index
)

st.session_state.current_page = selected_page
page = st.session_state.current_page

st.sidebar.divider()
st.sidebar.markdown("""
### 📊 How It Works
- **ML Model:** XGBoost Classifier
- **Features:** 9 statistical metrics
- **Training:** Historical match data
- **Accuracy:** Based on Elo + Form

### ⚙️ Features
- Real-time predictions
- Monte Carlo simulations
- Detailed team analytics
- Win probability trends
""")

# =====================================================
# PAGES
# =====================================================

if page == "🏠 Dashboard":
    st.markdown("# 🌍 FIFA World Cup 2026 Prediction Hub")
    
    st.markdown("""
    Welcome to the **ultimate World Cup prediction platform** powered by advanced machine learning.
    Use this tool to predict match outcomes, simulate tournaments, and analyze team performance.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Teams Analyzed", len(stats))
    
    with col2:
        st.metric("🧠 Model Type", "XGBoost")
    
    with col3:
        st.metric("📈 Features Used", "9")
    
    st.divider()
    
    # Top Contenders
    st.markdown("### 🏆 Top World Cup Contenders")
    
    top_teams_df = stats.nlargest(8, "elo")[["elo", "form", "goals_scored", "goals_conceded"]].copy()
    top_teams_df = top_teams_df.rename(columns={
        "elo": "ELO Rating",
        "form": "Form",
        "goals_scored": "Goals (Avg)",
        "goals_conceded": "Conceded (Avg)"
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(top_teams_df, use_container_width=True)
    
    with col2:
        fig = px.bar(
            top_teams_df.reset_index().rename(columns={"index": "Team"}),
            x="Team",
            y="ELO Rating",
            title="Top Teams by ELO Rating",
            color="ELO Rating",
            color_continuous_scale="Viridis",
            height=400
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("### 🎯 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚽ Go to Match Predictor", use_container_width=True, key="btn1"):
            st.session_state.current_page = "⚽ Match Predictor"
            st.rerun()
    
    with col2:
        if st.button("🏆 Run Tournament", use_container_width=True, key="btn2"):
            st.session_state.current_page = "🏆 Tournament Simulator"
            st.rerun()
    
    with col3:
        if st.button("📈 View Rankings", use_container_width=True, key="btn3"):
            st.session_state.current_page = "📈 Team Rankings"
            st.rerun()

# =====================================================
# MATCH PREDICTOR PAGE
# =====================================================

elif page == "⚽ Match Predictor":
    st.markdown("# ⚽ Match Predictor")
    st.markdown("Predict international football matches with AI-powered analytics")
    
    st.divider()
    
    teams = sorted(stats.index.tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "🏠 Home Team",
            teams,
            index=teams.index("Argentina") if "Argentina" in teams else 0
        )
    
    with col2:
        away_team = st.selectbox(
            "✈️ Away Team",
            teams,
            index=teams.index("Brazil") if "Brazil" in teams else 1
        )
    
    if home_team == away_team:
        st.warning("⚠️ Please select different teams")
    else:
        # Team Stats
        home = stats.loc[home_team]
        away = stats.loc[away_team]
        
        st.divider()
        st.markdown("### 📊 Team Comparison")
        
        comparison = pd.DataFrame({
            "Metric": ["Form", "Goals Scored (Avg)", "Goals Conceded (Avg)", "ELO Rating"],
            home_team: [
                f"{home['form']:.2f}",
                f"{home['goals_scored']:.2f}",
                f"{home['goals_conceded']:.2f}",
                f"{home['elo']:.0f}"
            ],
            away_team: [
                f"{away['form']:.2f}",
                f"{away['goals_scored']:.2f}",
                f"{away['goals_conceded']:.2f}",
                f"{away['elo']:.0f}"
            ]
        })
        
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            radar_data = {
                "Metric": ["Form", "Goals Scored", "Goals Conceded", "ELO"],
                home_team: [
                    home['form'] / 10,
                    home['goals_scored'],
                    home['goals_conceded'],
                    home['elo'] / 200
                ],
                away_team: [
                    away['form'] / 10,
                    away['goals_scored'],
                    away['goals_conceded'],
                    away['elo'] / 200
                ]
            }
            radar_df = pd.DataFrame(radar_data)
            
            fig = px.bar(radar_df, x="Metric", y=[home_team, away_team], barmode="group", title="Team Stats Comparison")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Head to Head Indicator
            elo_diff = home['elo'] - away['elo']
            strength_diff = abs(elo_diff)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=strength_diff,
                title="ELO Advantage",
                delta={'reference': 0, 'increasing': {'color': 'green'}},
                gauge={
                    'axis': {'range': [0, 500]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                        {'range': [100, 250], 'color': "gray"},
                        {'range': [250, 500], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 250
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Prediction Button
        if st.button("🔮 Predict Match Outcome", use_container_width=True, key="predict_btn"):
            with st.spinner("Analyzing match data..."):
                probs = predict_match(home_team, away_team, stats, model)
                
                away_win = round(probs[0] * 100, 2)
                draw = round(probs[1] * 100, 2)
                home_win = round(probs[2] * 100, 2)
                
                st.markdown("### 🎯 Match Prediction")
                
                # Main Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"{get_team_flag(home_team)} {home_team} Win",
                        f"{home_win:.2f}%",
                        delta=f"+{home_win - 33.33:.2f}% vs baseline"
                    )
                
                with col2:
                    st.metric(
                        "🤝 Draw",
                        f"{draw:.2f}%",
                        delta=f"{draw - 33.33:+.2f}% vs baseline"
                    )
                
                with col3:
                    st.metric(
                        f"{get_team_flag(away_team)} {away_team} Win",
                        f"{away_win:.2f}%",
                        delta=f"{away_win - 33.33:+.2f}% vs baseline"
                    )
                
                st.divider()
                
                # Confidence Level
                max_prob = max(probs) * 100
                confidence = get_confidence_level(max_prob)
                st.markdown(f"### {confidence}")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    chart_df = pd.DataFrame({
                        "Outcome": [home_team, "Draw", away_team],
                        "Probability": [home_win, draw, away_win],
                        "Color": ["#667eea", "#FFD700", "#764ba2"]
                    })
                    
                    fig = px.pie(
                        chart_df,
                        names="Outcome",
                        values="Probability",
                        title="Win Probability Distribution",
                        color_discrete_sequence=chart_df["Color"],
                        hole=0.4
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure(data=[
                        go.Bar(name=home_team, x=[home_team], y=[home_win], marker_color='#667eea'),
                        go.Bar(name='Draw', x=['Draw'], y=[draw], marker_color='#FFD700'),
                        go.Bar(name=away_team, x=[away_team], y=[away_win], marker_color='#764ba2')
                    ])
                    fig.update_layout(
                        title="Probability Comparison",
                        yaxis_title="Probability (%)",
                        barmode='group',
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Analysis
                st.divider()
                st.markdown("### 📈 Detailed Analysis")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown(f"""
                    **{home_team} Analysis:**
                    - ELO Rating: `{home['elo']:.0f}`
                    - Form Score: `{home['form']:.2f}`
                    - Goals/Match: `{home['goals_scored']:.2f}`
                    - Defense: `{home['goals_conceded']:.2f}` conceded/match
                    - Win Probability: **{home_win:.2f}%**
                    """)
                
                with analysis_col2:
                    st.markdown(f"""
                    **{away_team} Analysis:**
                    - ELO Rating: `{away['elo']:.0f}`
                    - Form Score: `{away['form']:.2f}`
                    - Goals/Match: `{away['goals_scored']:.2f}`
                    - Defense: `{away['goals_conceded']:.2f}` conceded/match
                    - Win Probability: **{away_win:.2f}%**
                    """)

# =====================================================
# TOURNAMENT SIMULATOR PAGE
# =====================================================

elif page == "🏆 Tournament Simulator":
    st.markdown("# 🏆 World Cup Tournament Simulator")
    st.markdown("Run Monte Carlo simulations to estimate championship odds")
    
    st.divider()
    
    # Tournament configuration
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
    
    with col2:
        top_teams = st.multiselect(
            "Select Teams for Tournament",
            sorted(stats.index.tolist()),
            default=["Argentina", "Brazil", "France", "Spain", "England", "Portugal", "Germany", "Netherlands"]
        )
    
    if len(top_teams) < 2:
        st.warning("⚠️ Please select at least 2 teams")
    else:
        st.divider()
        
        if st.button("🚀 Run Simulations", use_container_width=True):
            with st.spinner(f"Running {num_simulations} simulations..."):
                champions = simulate_tournament(top_teams, stats, model, num_sims=num_simulations)
                results = Counter(champions)
                
                simulation_df = pd.DataFrame({
                    "Team": list(results.keys()),
                    "Wins": list(results.values()),
                    "Probability (%)": [round(count / num_simulations * 100, 2) for count in results.values()]
                })
                
                simulation_df = simulation_df.sort_values("Probability (%)", ascending=False).reset_index(drop=True)
                
                st.markdown("### 🏆 Championship Odds")
                
                # Top Winners
                col1, col2, col3 = st.columns(3)
                
                if len(simulation_df) > 0:
                    with col1:
                        st.metric(
                            f"🥇 Champion: {simulation_df.iloc[0]['Team']}",
                            f"{simulation_df.iloc[0]['Probability (%)']:.2f}%"
                        )
                    
                    if len(simulation_df) > 1:
                        with col2:
                            st.metric(
                                f"🥈 Runner-up: {simulation_df.iloc[1]['Team']}",
                                f"{simulation_df.iloc[1]['Probability (%)']:.2f}%"
                            )
                    
                    if len(simulation_df) > 2:
                        with col3:
                            st.metric(
                                f"🥉 Third: {simulation_df.iloc[2]['Team']}",
                                f"{simulation_df.iloc[2]['Probability (%)']:.2f}%"
                            )
                
                st.divider()
                
                # Results Table
                st.markdown("### 📊 Full Results")
                st.dataframe(simulation_df, use_container_width=True, hide_index=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        simulation_df,
                        x="Team",
                        y="Probability (%)",
                        title="Championship Winning Probability",
                        color="Probability (%)",
                        color_continuous_scale="Viridis",
                        height=500
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        simulation_df.head(10),
                        names="Team",
                        values="Probability (%)",
                        title="Top 10 Teams Distribution",
                        hole=0.3
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TEAM RANKINGS PAGE
# =====================================================

elif page == "📈 Team Rankings":
    st.markdown("# 📈 Team Rankings & Statistics")
    
    st.divider()
    
    # Ranking by ELO
    rankings = stats.sort_values("elo", ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔝 Top Team", rankings.index[0], f"ELO: {rankings.iloc[0]['elo']:.0f}")
    
    with col2:
        st.metric("📊 Total Teams", len(rankings))
    
    with col3:
        st.metric("⌀ Average ELO", f"{rankings['elo'].mean():.0f}")
    
    st.divider()
    
    # Filter and display
    col1, col2 = st.columns(2)
    
    with col1:
        top_n = st.slider("Show Top N Teams", min_value=5, max_value=len(rankings), value=20)
    
    with col2:
        sort_by = st.selectbox("Sort By", ["ELO Rating", "Form", "Goals Scored", "Goals Conceded"])
    
    # Sort by selected metric
    sort_column = {
        "ELO Rating": "elo",
        "Form": "form",
        "Goals Scored": "goals_scored",
        "Goals Conceded": "goals_conceded"
    }[sort_by]
    
    display_rankings = stats.sort_values(sort_column, ascending=False).head(top_n)
    display_rankings = display_rankings[["elo", "form", "goals_scored", "goals_conceded"]].copy()
    display_rankings.columns = ["ELO", "Form", "Goals (Avg)", "Conceded (Avg)"]
    
    st.dataframe(display_rankings, use_container_width=True)
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            display_rankings.head(15).reset_index().rename(columns={"index": "Team"}),
            x="Team",
            y="ELO",
            title="Top 15 Teams by ELO Rating",
            color="ELO",
            color_continuous_scale="Plasma",
            height=500
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            display_rankings.reset_index().rename(columns={"index": "Team"}),
            x="Goals (Avg)",
            y="Conceded (Avg)",
            size="ELO",
            color="ELO",
            hover_name="Team",
            title="Team Offensive vs Defensive Performance",
            color_continuous_scale="Viridis",
            height=500
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "📊 Analytics":
    st.markdown("# 📊 Advanced Analytics")
    st.markdown("Deep dive into team statistics and model insights")
    
    st.divider()
    
    selected_team = st.selectbox("Select Team for Analysis", sorted(stats.index.tolist()))
    
    if selected_team:
        team_data = stats.loc[selected_team]
        
        st.markdown(f"## {get_team_flag(selected_team)} {selected_team}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ELO Rating", f"{team_data['elo']:.0f}")
        
        with col2:
            st.metric("Form Score", f"{team_data['form']:.2f}")
        
        with col3:
            st.metric("Goals/Match", f"{team_data['goals_scored']:.2f}")
        
        with col4:
            st.metric("Defense", f"{team_data['goals_conceded']:.2f}")
        
        st.divider()
        
        # Team stats visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance gauge
            form_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=team_data['form'],
                title="Form Score",
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgray"},
                        {'range': [3, 6], 'color': "gray"},
                        {'range': [6, 10], 'color': "darkgray"}
                    ]
                }
            ))
            form_gauge.update_layout(height=400)
            st.plotly_chart(form_gauge, use_container_width=True)
        
        with col2:
            # Win/Loss balance
            fig = go.Figure(data=[
                go.Bar(name='Goals For', x=['Performance'], y=[team_data['goals_scored']], marker_color='#667eea'),
                go.Bar(name='Goals Against', x=['Performance'], y=[team_data['goals_conceded']], marker_color='#764ba2')
            ])
            fig.update_layout(
                barmode='group',
                title="Offensive vs Defensive Stats",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Head-to-Head predictions
        st.markdown("### 🔄 Head-to-Head Predictions")
        
        opponents = st.multiselect(
            "Select Opponents to Compare",
            [t for t in sorted(stats.index.tolist()) if t != selected_team],
            default=[t for t in sorted(stats.index.tolist())[:3] if t != selected_team]
        )
        
        if opponents:
            h2h_data = []
            
            for opponent in opponents:
                probs = predict_match(selected_team, opponent, stats, model)
                h2h_data.append({
                    "Opponent": opponent,
                    "Win %": round(probs[2] * 100, 2),
                    "Draw %": round(probs[1] * 100, 2),
                    "Loss %": round(probs[0] * 100, 2)
                })
            
            h2h_df = pd.DataFrame(h2h_data)
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = px.bar(
                h2h_df,
                x="Opponent",
                y=["Win %", "Draw %", "Loss %"],
                title=f"{selected_team} Head-to-Head Probabilities",
                barmode="stack",
                color_discrete_sequence=["#667eea", "#FFD700", "#764ba2"]
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("""
<div style='text-align: center'>
    <small>FIFA World Cup 2026 Predictor</small><br>
    <small>Powered by XGBoost ML</small><br>
    <small>© 2026 All Rights Reserved</small>
</div>
""", unsafe_allow_html=True)