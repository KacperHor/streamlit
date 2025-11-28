import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Konfiguracja strony
st.set_page_config(page_title="Analiza Win i Parowania Jedzenia", layout="wide")

st.title("🍷 Analiza Danych o Winie i Parowaniu Potraw")
st.markdown("""
Aplikacja wizualizuje połączenia między winem a jedzeniem oraz analizuje chemiczne właściwości wina wpływające na jego jakość.
Wykorzystuje techniki wizualizacji danych oraz modele uczenia maszynowego (Random Forest).
""")

# Funkcja ładowania danych
@st.cache_data
def load_data():
    # Wczytywanie danych z obsługą błędów
    try:
        pairings = pd.read_csv('wine_food_pairings.csv')
        quality = pd.read_csv('winequality-red.csv')
        return pairings, quality
    except FileNotFoundError:
        st.error("Nie znaleziono plików CSV. Upewnij się, że 'wine_food_pairings.csv' i 'winequality-red.csv' są w katalogu roboczym.")
        return None, None

df_pairings, df_quality = load_data()

if df_pairings is not None and df_quality is not None:
    
    # Zakładki dla lepszej organizacji
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Przegląd Danych", "gourmet Parowanie Win", "🧪 Chemia Wina", "🤖 Modele ML (Random Forest)"])

    # --- TAB 1: PRZEGLĄD DANYCH ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Zbiór: Parowanie Win (Wine Food Pairings)")
            st.dataframe(df_pairings.head())
            st.write(f"Wymiary: {df_pairings.shape}")
        with col2:
            st.subheader("Zbiór: Jakość Czerwonego Wina")
            st.dataframe(df_quality.head())
            st.write(f"Wymiary: {df_quality.shape}")

    # --- TAB 2: PAROWANIE WIN (Wizualizacje) ---
    with tab2:
        st.header("Analiza połączeń: Wino - Jedzenie")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rozkład oceny parowania")
            fig, ax = plt.subplots()
            sns.countplot(data=df_pairings, x='pairing_quality', palette='viridis', ax=ax)
            ax.set_title("Liczba parowań wg oceny jakości")
            ax.set_xlabel("Ocena jakości (1-5)")
            ax.set_ylabel("Liczba wystąpień")
            st.pyplot(fig)

        with col2:
            st.subheader("Średnia jakość parowania: Kategoria Wina vs Kategoria Jedzenia")
            # Pivot table dla heatmapy
            pivot_table = df_pairings.pivot_table(index='wine_category', columns='food_category', values='pairing_quality', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".1f", ax=ax)
            ax.set_title("Heatmapa jakości parowania")
            st.pyplot(fig)

        st.subheader("Najlepsze wina dla wybranej kuchni")
        selected_cuisine = st.selectbox("Wybierz kuchnię:", df_pairings['cuisine'].unique())
        
        best_pairings = df_pairings[df_pairings['cuisine'] == selected_cuisine].sort_values(by='pairing_quality', ascending=False).head(5)
        st.table(best_pairings[['wine_type', 'food_item', 'pairing_quality', 'description']])

    # --- TAB 3: CHEMIA WINA (Wizualizacje) ---
    with tab3:
        st.header("Analiza właściwości fizykochemicznych wina")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Macierz korelacji cech")
            corr = df_quality.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar=True)
            st.pyplot(fig)

        with col2:
            st.markdown("### Wnioski z korelacji")
            st.info("""
            Z macierzy korelacji możemy odczytać, które cechy najmocniej wpływają na jakość ('quality'):
            - **Alcohol**: Często ma silną pozytywną korelację z jakością.
            - **Volatile acidity**: Zazwyczaj ma negatywną korelację (kwas octowy psuje smak).
            """)

        st.subheader("Wpływ zawartości alkoholu na jakość")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='quality', y='alcohol', data=df_quality, palette='Blues', ax=ax)
        ax.set_title("Zawartość alkoholu w zależności od oceny jakości")
        st.pyplot(fig)

    # --- TAB 4: MODELE ML (Random Forest) ---
    with tab4:
        st.header("Modelowanie Predykcyjne")

        # MODEL 1: Przewidywanie jakości parowania (Klasyfikacja)
        st.subheader("1. Przewidywanie sukcesu parowania (Random Forest Classifier)")
        st.markdown("Model uczy się na podstawie typu wina, kategorii jedzenia i kuchni, aby przewidzieć ocenę (1-5).")

        # Preprocessing
        le_dict = {}
        df_ml_pair = df_pairings.copy()
        categorical_cols = ['wine_type', 'wine_category', 'food_item', 'food_category', 'cuisine', 'quality_label']
        
        # Kodowanie zmiennych kategorycznych
        for col in categorical_cols:
            if col in df_ml_pair.columns:
                le = LabelEncoder()
                df_ml_pair[col] = le.fit_transform(df_ml_pair[col].astype(str))
                le_dict[col] = le

        X_pair = df_ml_pair[['wine_type', 'wine_category', 'food_category', 'cuisine']]
        y_pair = df_ml_pair['pairing_quality']

        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)

        if st.button("Trenuj Model Parowania"):
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train_p, y_train_p)
            y_pred_p = rf_clf.predict(X_test_p)

            acc = accuracy_score(y_test_p, y_pred_p)
            st.success(f"Dokładność modelu (Accuracy): {acc:.2%}")

            # Feature Importance
            st.markdown("**Ważność cech w parowaniu:**")
            feature_imp = pd.DataFrame({'Cecha': X_pair.columns, 'Waga': rf_clf.feature_importances_}).sort_values('Waga', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Waga', y='Cecha', data=feature_imp, palette='viridis', ax=ax)
            ax.set_title("Co decyduje o dobrym połączeniu?")
            st.pyplot(fig)

        st.divider()

        # MODEL 2: Przewidywanie jakości wina (Regresja)
        st.subheader("2. Przewidywanie jakości wina na podstawie składu (Random Forest Regressor)")
        st.markdown("Model przewiduje dokładną ocenę punktową wina na podstawie parametrów chemicznych.")

        X_qual = df_quality.drop('quality', axis=1)
        y_qual = df_quality['quality']

        X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_qual, y_qual, test_size=0.2, random_state=42)

        if st.button("Trenuj Model Jakości Wina"):
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train_q, y_train_q)
            y_pred_q = rf_reg.predict(X_test_q)

            mse = mean_squared_error(y_test_q, y_pred_q)
            r2 = r2_score(y_test_q, y_pred_q)

            col1, col2 = st.columns(2)
            col1.metric("Błąd średniokwadratowy (MSE)", f"{mse:.3f}")
            col2.metric("Współczynnik R2", f"{r2:.3f}")

            # Feature Importance
            st.markdown("**Który składnik chemiczny jest najważniejszy?**")
            feature_imp_q = pd.DataFrame({'Cecha': X_qual.columns, 'Waga': rf_reg.feature_importances_}).sort_values('Waga', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Waga', y='Cecha', data=feature_imp_q, palette='magma', ax=ax)
            ax.set_title("Ważność cech chemicznych dla jakości wina")
            st.pyplot(fig)

else:
    st.warning("Oczekiwanie na dane...")
