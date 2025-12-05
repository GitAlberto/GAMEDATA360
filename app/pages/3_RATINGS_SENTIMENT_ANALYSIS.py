import streamlit as st
import pandas as pd
import altair as alt
import ast
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path

PROFESSIONAL_COLORS = {
    'primary': '#2563eb',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'neutral': '#6b7280',
    'positive': '#059669',
    'negative': '#dc2626',
    'background': '#f8fafc',
}

PROFESSIONAL_TEMPLATE = {
    'layout': go.Layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        colorway=['#2563eb', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'],
    )
}

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("⚠️ Le module 'wordcloud' n'est pas installé. Le word cloud ne sera pas disponible.")

st.set_page_config(page_title="GameData360 — QUALITÉ & SATISFACTION", layout="wide")
st.title("⿣ QUALITÉ & SATISFACTION — Ratings & Sentiment Analysis")

@st.cache_data
def load_data():
    """Charge et prépare les données avec mise en cache"""
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "nettoyes" / "jeux_analysis_final.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        possible_paths = [
            base_path / "data" / "nettoyes" / "jeux_analysis_final.csv",
            Path("data/nettoyes/jeux_analysis_final.csv"),
            Path("../../data/nettoyes/jeux_analysis_final.csv"),
        ]
        
        df = None
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                break
        
        if df is None:
            raise FileNotFoundError(
                f"Fichier de données introuvable. Cherché dans: {[str(p) for p in possible_paths]}"
            )
    
    for col in ["Genres", "Categories", "Tags"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    
    return df

df_analyse = load_data()

st.header("🔎 Filtres généraux")

@st.cache_data
def get_unique_values(df):
    """Extrait les valeurs uniques pour les filtres"""
    unique_genres = []
    unique_cats = []
    unique_tags = []
    
    if "Genres" in df.columns:
        df_genres = df.explode("Genres")
        unique_genres = sorted(df_genres["Genres"].dropna().astype(str).str.strip().str.lower().unique().tolist())
    
    if "Categories" in df.columns:
        df_cats = df.explode("Categories")
        unique_cats = sorted(df_cats["Categories"].dropna().astype(str).str.strip().str.lower().unique().tolist())
    
    if "Tags" in df.columns:
        df_tags = df.explode("Tags")
        unique_tags = sorted(df_tags["Tags"].dropna().astype(str).str.strip().str.lower().unique().tolist())
    
    return unique_genres, unique_cats, unique_tags

unique_genres, unique_cats, unique_tags = get_unique_values(df_analyse)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    selected_genres = st.multiselect("🎭 Genres", unique_genres)
with col_f2:
    selected_categories = st.multiselect("📂 Catégories", unique_cats)
with col_f3:
    selected_tags = st.multiselect("🏷️ Tags", unique_tags)

if st.button("🔄 Réinitialiser les filtres"):
    selected_genres = []
    selected_categories = []
    selected_tags = []
    st.rerun()

@st.cache_data
def apply_filters(df, selected_genres, selected_categories, selected_tags):
    """Applique les filtres de manière optimisée"""
    df_filtered = df.copy()
    
    if selected_genres:
        df_filtered = df_filtered[
            df_filtered["Genres"].apply(
                lambda lst: any(g in [x.lower() for x in lst] for g in selected_genres)
            )
        ]
    
    if selected_categories:
        df_filtered = df_filtered[
            df_filtered["Categories"].apply(
                lambda lst: any(c in [x.lower() for x in lst] for c in selected_categories)
            )
        ]
    
    if selected_tags:
        df_filtered = df_filtered[
            df_filtered["Tags"].apply(
                lambda lst: any(t in [x.lower() for x in lst] for t in selected_tags)
            )
        ]
    
    return df_filtered

df_filtered = apply_filters(df_analyse, selected_genres, selected_categories, selected_tags)

required_cols = ["User score", "Metacritic score", "Positive", "Negative", "Genres", "Tags", "Name"]
missing_cols = [col for col in required_cols if col not in df_filtered.columns]

if missing_cols:
    st.error(f"⚠️ Colonnes manquantes dans le dataset: {', '.join(missing_cols)}")
    st.stop()

df_filtered = df_filtered[
    (df_filtered["User score"] > 0) | (df_filtered["Metacritic score"] > 0)
].copy()

df_filtered["Total Reviews"] = df_filtered["Positive"] + df_filtered["Negative"]
df_filtered["Positive Ratio"] = df_filtered["Positive"] / df_filtered["Total Reviews"].replace(0, 1)
df_filtered["Negative Ratio"] = df_filtered["Negative"] / df_filtered["Total Reviews"].replace(0, 1)

st.header("📊 Graphiques")

st.subheader("📦 Boxplot du score par genre")

@st.cache_data
def prepare_boxplot_data(df_filtered):
    """Prépare les données pour le boxplot"""
    df_boxplot = df_filtered[df_filtered["User score"] > 0].copy()
    if df_boxplot.empty:
        return None, None
    
    df_boxplot = df_boxplot.explode("Genres")
    df_boxplot = df_boxplot[df_boxplot["Genres"].notna()].copy()
    df_boxplot["Genre"] = df_boxplot["Genres"].astype(str).str.strip()
    
    genre_counts = df_boxplot["Genre"].value_counts().head(15)
    df_boxplot = df_boxplot[df_boxplot["Genre"].isin(genre_counts.index)]
    
    df_genre_scores = pd.DataFrame({
        "Genre": df_boxplot["Genre"],
        "User Score": df_boxplot["User score"],
        "Metacritic Score": df_boxplot["Metacritic score"].where(df_boxplot["Metacritic score"] > 0)
    })
    
    return df_genre_scores, genre_counts

df_genre_scores, genre_counts = prepare_boxplot_data(df_filtered)

if df_genre_scores is not None and not df_genre_scores.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        df_user = df_genre_scores[df_genre_scores["User Score"] > 0]
        if not df_user.empty:
            fig_user = px.box(
                df_user,
                x="Genre",
                y="User Score",
                title="User Score par Genre",
                labels={"User Score": "User Score", "Genre": "Genre"},
                color_discrete_sequence=[PROFESSIONAL_COLORS['primary']]
            )
            fig_user.update_layout(
                template='plotly_white',
                font=dict(family="Inter, sans-serif", size=11),
                showlegend=False,
                height=400
            )
            fig_user.update_xaxes(tickangle=-45, showgrid=True, gridcolor='#e5e7eb')
            fig_user.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
            st.plotly_chart(fig_user, use_container_width=True)
    
    with col2:
        df_meta = df_genre_scores[df_genre_scores["Metacritic Score"].notna()]
        if not df_meta.empty:
            fig_meta = px.box(
                df_meta,
                x="Genre",
                y="Metacritic Score",
                title="Metacritic Score par Genre",
                labels={"Metacritic Score": "Metacritic Score", "Genre": "Genre"},
                color_discrete_sequence=[PROFESSIONAL_COLORS['secondary']]
            )
            fig_meta.update_layout(
                template='plotly_white',
                font=dict(family="Inter, sans-serif", size=11),
                showlegend=False,
                height=400
            )
            fig_meta.update_xaxes(tickangle=-45, showgrid=True, gridcolor='#e5e7eb')
            fig_meta.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
            st.plotly_chart(fig_meta, use_container_width=True)

st.subheader("📊 Ratio reviews positives / négatives par genre")

@st.cache_data
def prepare_ratio_data(df_filtered):
    """Prépare les données pour le ratio"""
    df_ratio = df_filtered[df_filtered["Total Reviews"] > 0].copy()
    if df_ratio.empty:
        return None
    
    df_ratio = df_ratio.explode("Genres")
    df_ratio = df_ratio[df_ratio["Genres"].notna()].copy()
    df_ratio["Genre"] = df_ratio["Genres"].astype(str).str.strip()
    
    df_genre_sentiment = pd.DataFrame({
        "Genre": df_ratio["Genre"],
        "Positive": df_ratio["Positive"],
        "Negative": df_ratio["Negative"],
        "Total": df_ratio["Total Reviews"]
    })
    df_genre_agg = df_genre_sentiment.groupby("Genre").agg({
        "Positive": "sum",
        "Negative": "sum",
        "Total": "sum"
    }).reset_index()
    
    df_genre_agg["Positive Ratio"] = df_genre_agg["Positive"] / df_genre_agg["Total"]
    df_genre_agg["Negative Ratio"] = df_genre_agg["Negative"] / df_genre_agg["Total"]
    df_genre_agg = df_genre_agg.sort_values("Total", ascending=False).head(15)
    
    return df_genre_agg

df_genre_agg = prepare_ratio_data(df_filtered)

if df_genre_agg is not None and not df_genre_agg.empty:
    fig_ratio = go.Figure()
    
    fig_ratio.add_trace(go.Bar(
        x=df_genre_agg["Genre"],
        y=df_genre_agg["Positive Ratio"] * 100,
        name="Positives",
        marker_color=PROFESSIONAL_COLORS['positive'],
        hovertemplate='<b>%{x}</b><br>Positives: %{y:.1f}%<extra></extra>'
    ))
    
    fig_ratio.add_trace(go.Bar(
        x=df_genre_agg["Genre"],
        y=df_genre_agg["Negative Ratio"] * 100,
        name="Négatives",
        marker_color=PROFESSIONAL_COLORS['negative'],
        hovertemplate='<b>%{x}</b><br>Négatives: %{y:.1f}%<extra></extra>'
    ))
    
    fig_ratio.update_layout(
        title=dict(text="Ratio Reviews Positives/Négatives par Genre", font=dict(size=16)),
        xaxis_title="Genre",
        yaxis_title="Pourcentage (%)",
        barmode="stack",
        xaxis_tickangle=-45,
        height=450,
        template='plotly_white',
        font=dict(family="Inter, sans-serif", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_ratio.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig_ratio.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    st.plotly_chart(fig_ratio, use_container_width=True)

st.subheader("🎯 User score vs Metacritic score")

df_scatter = df_filtered[
    (df_filtered["User score"] > 0) & (df_filtered["Metacritic score"] > 0)
].copy()

if not df_scatter.empty:
    if len(df_scatter) > 5000:
        df_scatter = df_scatter.sample(n=5000, random_state=42)
    
    df_scatter["Bubble Size"] = np.log1p(df_scatter["Total Reviews"]) * 5
    
    fig_scatter = px.scatter(
        df_scatter,
        x="Metacritic score",
        y="User score",
        size="Bubble Size",
        hover_data=["Name", "Positive", "Negative"],
        title="User Score vs Metacritic Score",
        labels={
            "Metacritic score": "Metacritic Score",
            "User score": "User Score"
        },
        trendline="ols",
        color_discrete_sequence=[PROFESSIONAL_COLORS['primary']]
    )
    
    fig_scatter.update_traces(
        marker=dict(
            opacity=0.5,
            line=dict(width=0.5, color='white')
        )
    )
    fig_scatter.update_layout(
        template='plotly_white',
        font=dict(family="Inter, sans-serif", size=11),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_scatter.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig_scatter.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    correlation = df_scatter["User score"].corr(df_scatter["Metacritic score"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Corrélation", f"{correlation:.3f}", help="Corrélation entre User Score et Metacritic Score")

st.subheader("☁️ Word cloud des tags liés à une note élevée")

if WORDCLOUD_AVAILABLE:
    score_threshold = df_filtered["User score"].quantile(0.75)
    
    df_wc = df_filtered[df_filtered["User score"] >= score_threshold].copy()
    df_wc = df_wc.explode("Tags")
    df_wc = df_wc[df_wc["Tags"].notna()].copy()
    df_wc["Tag"] = df_wc["Tags"].astype(str).str.strip().str.lower()
    df_wc["Weight"] = (df_wc["User score"] / 100 * 10 + 1).astype(int)
    
    if not df_wc.empty:
        tags_repeated = []
        for tag, weight in zip(df_wc["Tag"], df_wc["Weight"]):
            tags_repeated.extend([tag] * int(weight))
        
        if tags_repeated:
            tag_counts = Counter(tags_repeated)
            top_tags = dict(tag_counts.most_common(80))
            
            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                max_words=80,
                colormap="Blues",
                relative_scaling=0.5,
                min_font_size=10,
                max_font_size=120
            ).generate_from_frequencies(top_tags)
            
            fig_wc, ax = plt.subplots(figsize=(14, 7), facecolor='white')
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Tags les plus associés aux jeux avec scores élevés", 
                        fontsize=16, pad=20, fontweight='600', color='#1f2937')
            plt.tight_layout()
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
    else:
        st.info("Pas assez de données pour générer le word cloud")
else:
    st.error("Le module 'wordcloud' n'est pas installé. Veuillez l'installer avec: pip install wordcloud")

st.subheader("🔥 Heatmap Genres × Sentiment")

df_hm = df_filtered[df_filtered["Total Reviews"] > 0].copy()
df_hm = df_hm.explode("Genres")
df_hm = df_hm[df_hm["Genres"].notna()].copy()
df_hm["Genre"] = df_hm["Genres"].astype(str).str.strip()

if not df_hm.empty:
    df_heatmap = pd.DataFrame({
        "Genre": df_hm["Genre"],
        "Positive": df_hm["Positive"],
        "Negative": df_hm["Negative"],
        "Total": df_hm["Total Reviews"]
    })
    df_heatmap_agg = df_heatmap.groupby("Genre").agg({
        "Positive": "sum",
        "Negative": "sum",
        "Total": "sum"
    }).reset_index()
    
    df_heatmap_agg["Sentiment Score"] = (df_heatmap_agg["Positive"] - df_heatmap_agg["Negative"]) / df_heatmap_agg["Total"]
    df_heatmap_agg = df_heatmap_agg[df_heatmap_agg["Total"] >= 100].sort_values("Total", ascending=False).head(12)
    
    heatmap_data = df_heatmap_agg.set_index("Genre")[["Sentiment Score"]].T
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Genre", y="Sentiment Score", color="Score"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale=[[0, '#dc2626'], [0.5, '#fbbf24'], [1, '#10b981']],
        aspect="auto",
        title="Heatmap Sentiment par Genre",
        text_auto='.2f'
    )
    
    fig_heatmap.update_layout(
        template='plotly_white',
        font=dict(family="Inter, sans-serif", size=11),
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_heatmap.update_xaxes(tickangle=-45, showgrid=False)
    fig_heatmap.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.header("📄 Table")

st.subheader("🎮 Jeux triés par score / sentiment")

sort_option = st.selectbox(
    "Trier par:",
    [
        "User Score (décroissant)",
        "Metacritic Score (décroissant)",
        "Ratio Positif (décroissant)",
        "Total Reviews (décroissant)",
        "Score Moyen (User + Metacritic)"
    ]
)

df_table = df_filtered.copy()
df_table["Score Moyen"] = (
    df_table["User score"].fillna(0) + df_table["Metacritic score"].fillna(0)
) / 2

if sort_option == "User Score (décroissant)":
    df_table = df_table.sort_values("User score", ascending=False)
elif sort_option == "Metacritic Score (décroissant)":
    df_table = df_table.sort_values("Metacritic score", ascending=False)
elif sort_option == "Ratio Positif (décroissant)":
    df_table = df_table.sort_values("Positive Ratio", ascending=False)
elif sort_option == "Total Reviews (décroissant)":
    df_table = df_table.sort_values("Total Reviews", ascending=False)
elif sort_option == "Score Moyen (User + Metacritic)":
    df_table = df_table.sort_values("Score Moyen", ascending=False)

display_cols = ["Name", "User score", "Metacritic score", "Positive", "Negative", 
                "Positive Ratio", "Total Reviews"]

df_display = df_table[display_cols].copy()
df_display["Positive Ratio"] = (df_display["Positive Ratio"] * 100).round(2)
df_display = df_display.rename(columns={
    "Name": "Nom du Jeu",
    "User score": "User Score",
    "Metacritic score": "Metacritic Score",
    "Positive": "Reviews Positives",
    "Negative": "Reviews Négatives",
    "Positive Ratio": "Ratio Positif (%)",
    "Total Reviews": "Total Reviews"
})

st.dataframe(
    df_display.head(100),
    use_container_width=True,
    height=400
)

st.caption(f"Affichage des 100 premiers jeux sur {len(df_table)} au total")

