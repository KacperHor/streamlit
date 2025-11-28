import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Wine & Food Connections", layout="wide", page_icon="🍷")

# --- ŁADOWANIE DANYCH ---
@st.cache_data
def load_data():
    try:
        # Wczytujemy pliki zakładając, że są w tym samym folderze co skrypt (struktura GitHub)
        df_pair = pd.read_csv('wine_food_pairings.csv')
        df_qual = pd.read_csv('winequality-red.csv')
        return df_pair, df_qual
    except FileNotFoundError as e:
        st.error(f"Nie znaleziono pliku! Błąd: {e}")
        return None, None

df_pairings, df_quality = load_data()

# Jeśli dane się nie wczytają, przerywamy
if df_pairings is None or df_quality is None:
    st.stop()

st.title("🍷 Wizualizacja Połączeń: Wino i Jedzenie")
st.markdown("Aplikacja analizuje, jak cechy chemiczne wpływają na jakość wina oraz wizualizuje **połączenia** między winem a potrawami.")

# --- ZAKŁADKI ---
tab1, tab2, tab3 = st.tabs(["🔗 Połączenia (Sankey & Network)", "🧪 Chemia Wina (Korelacje)", "🤖 AI: Random Forest"])

# ==========================================
# TAB 1: POŁĄCZENIA (WIZUALIZACJA RELACJI)
# ==========================================
with tab1:
    st.header("Jak wino łączy się z jedzeniem?")
    
    # 1. WYKRES SANKEY'A (Diagram przepływu)
    st.subheader("Przepływ: Kategoria Wina ➡️ Kategoria Jedzenia")
    st.caption("Ten wykres pokazuje, które kategorie win najczęściej są parowane z jakimi kategoriami jedzenia. Grubość linii oznacza średnią ocenę parowania.")
    
    # Przygotowanie danych do Sankey
    # Agregujemy dane: Źródło (Wino) -> Cel (Jedzenie) -> Wartość (Liczba dobrych parowań lub średnia ocena)
    sankey_data = df_pairings.groupby(['wine_category', 'food_category'])['pairing_quality'].mean().reset_index()
    
    # Tworzenie list unikalnych etykiet
    all_nodes = list(pd.concat([sankey_data['wine_category'], sankey_data['food_category']]).unique())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    source_indices = sankey_data['wine_category'].map(node_map)
    target_indices = sankey_data['food_category'].map(node_map)
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="rgba(255,0,0,0.6)"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=sankey_data['pairing_quality'], # Grubość linii to jakość parowania
            color="rgba(200, 200, 200, 0.4)"
        )
    )])
    fig_sankey.update_layout(title_text="Diagram połączeń (Sankey)", font_size=12, height=500)
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.divider()

    # 2. INTERAKTYWNA HEATMAPA
    st.subheader("Szczegółowa mapa połączeń (Heatmapa)")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Pivot table
        pivot = df_pairings.pivot_table(index='wine_type', columns='cuisine', values='pairing_quality', aggfunc='mean')
        fig_heat = px.imshow(pivot, 
                             labels=dict(x="Kuchnia", y="Szczep Wina", color="Ocena"),
                             x=pivot.columns, 
                             y=pivot.index,
                             color_continuous_scale='RdYlGn',
                             aspect="auto")
        fig_heat.update_layout(title="Jakość parowania: Szczep wina vs Kuchnia świata")
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with col2:
        st.info("💡 **Jak czytać:**\n\nZielone pola oznaczają wyśmienite połączenia. Czerwone to te, których należy unikać.\n\nMożesz przybliżać i oddalać wykres.")

# ==========================================
# TAB 2: CHEMIA WINA
# ==========================================
with tab2:
    st.header("Analiza Chemiczna Czerwonego Wina")
    
    # Wybór parametrów do scatter plot
    st.write("Zbadaj zależność między dwoma składnikami a jakością:")
    c1, c2, c3 = st.columns(3)
    x_axis = c1.selectbox("Oś X", df_quality.columns[:-1], index=10) # domyślnie alcohol
    y_axis = c2.selectbox("Oś Y", df_quality.columns[:-1], index=1)  # domyślnie volatile acidity
    color_by = c3.selectbox("Kolor (Grupa)", ['quality'])

    fig_scatter = px.scatter(df_quality, x=x_axis, y=y_axis, color=color_by, 
                             size='total sulfur dioxide', hover_data=['pH', 'density'],
                             color_continuous_scale='Bluered', title=f"Relacja: {x_axis} vs {y_axis}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Macierz korelacji
    st.subheader("Co najbardziej wpływa na jakość?")
    corr = df_quality.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 3: MACHINE LEARNING
# ==========================================
with tab3:
    st.header("Modele Random Forest 🌳")
    
    col_ml1, col_ml2 = st.columns(2)
    
    # --- MODEL 1: KLASYFIKACJA PAROWANIA ---
    with col_ml1:
        st.subheader("1. Predykcja: Czy to pasuje?")
        st.markdown("Model uczy się na podstawie danych historycznych, co do siebie pasuje.")
        
        # Prosty preprocessing - zamiana tekstu na liczby
        df_ml = df_pairings.copy()
        for col in ['wine_category', 'food_category', 'cuisine', 'wine_type', 'food_item']:
            df_ml[col] = df_ml[col].astype('category').cat.codes
            
        X = df_ml[['wine_category', 'food_category', 'cuisine']]
        y = df_ml['pairing_quality']
        
        if st.button("Trenuj Model Parowania", key="btn_pair"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_train)
            acc = accuracy_score(y_test, rf.predict(X_test))
            
            st.success(f"Model wytrenowany! Dokładność: {acc:.1%}")
            
            # Feature Importance
            imp = pd.DataFrame({'Cecha': X.columns, 'Waga': rf.feature_importances_})
            fig_imp = px.bar(imp, x='Cecha', y='Waga', title="Co jest kluczowe w parowaniu?")
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- MODEL 2: REGRESJA JAKOŚCI WINA ---
    with col_ml2:
        st.subheader("2. Predykcja: Jakość Wina")
        st.markdown("Analiza parametrów chemicznych w celu oceny punktowej.")
        
        X_q = df_quality.drop('quality', axis=1)
        y_q = df_quality['quality']
        
        if st.button("Trenuj Model Jakości", key="btn_qual"):
            X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_q, y_q, test_size=0.2, random_state=42)
            rf_reg = RandomForestRegressor(n_estimators=100)
            rf_reg.fit(X_train_q, y_train_q)
            mse = mean_squared_error(y_test_q, rf_reg.predict(X_test_q))
            
            st.success(f"Model gotowy! MSE (Błąd): {mse:.3f}")
            
            # Feature Importance
            imp_q = pd.DataFrame({'Cecha': X_q.columns, 'Waga': rf_reg.feature_importances_}).sort_values('Waga', ascending=False)
            fig_imp_q = px.bar(imp_q, x='Waga', y='Cecha', orientation='h', title="Najważniejsze składniki chemiczne")
            st.plotly_chart(fig_imp_q, use_container_width=True)
