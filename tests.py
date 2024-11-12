import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Fonction pour saisir les données de l'utilisateur
def get_user_input_data():
    st.subheader("Enter Data")
    data1 = st.text_area("Enter data for Sample 1 (separate values with commas):", '1,2,3,4,5')
    data2 = st.text_area("Enter data for Sample 2 (separate values with commas):", '2,3,4,5,6')

    # Convertir les chaînes saisies en listes de nombres
    data1 = [float(i) for i in data1.split(',')]
    data2 = [float(i) for i in data2.split(',')]
    
    return np.array(data1), np.array(data2)

# Fonction pour effectuer un t-test
def perform_t_test(data1, data2):
    st.subheader("T-Test Result")
    
    # Exécution du t-test
    t_stat, p_val = stats.ttest_ind(data1, data2)

    st.write("T-statistic:", t_stat)
    st.write("P-value:", p_val)

# Fonction pour effectuer un z-test
def perform_z_test(data1, data2):
    st.subheader("Z-Test Result")
    
    # Calcul de la moyenne et de l'écart-type
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

    # Taille des échantillons
    n1, n2 = len(data1), len(data2)

    # Exécution du z-test
    z_stat = (mean1 - mean2) / np.sqrt(std1**2/n1 + std2**2/n2)
    p_val = stats.norm.sf(abs(z_stat)) * 2

    st.write("Z-statistic:", z_stat)
    st.write("P-value:", p_val)

# Fonction pour effectuer le test du chi carré
def perform_chi_square_test(data1, data2):
    st.subheader("Chi-Square Test Result")
    
    # Création du dataframe à partir des données
    df = pd.DataFrame({'Sample 1': data1, 'Sample 2': data2})

    # Création d'une table de contingence
    contingency_table = pd.crosstab(df['Sample 1'], df['Sample 2'])

    # Exécution du test du chi carré
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)

    st.write("Chi-Square Value:", chi2)
    st.write("P-value:", p)

# Fonction pour effectuer la régression linéaire
def perform_linear_regression(data1, data2):
    st.subheader("Linear Regression Result")

    # Exécution de la régression linéaire
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)

    st.write("Slope:", slope)
    st.write("Intercept:", intercept)
    st.write("R-squared:", r_value**2)
    st.write("P-value:", p_value)

    # Tracé du graphe de régression linéaire
    plt.scatter(data1, data2, label='Data Points')
    plt.plot(data1, intercept + slope * data1, 'r', label='Linear Regression')
    plt.xlabel('Sample 1')
    plt.ylabel('Sample 2')
    plt.legend()
    st.pyplot(plt)

# Fonction principale de l'application
def perform_tests():
    st.title("Statistical Tests with User Data")

    data1, data2 = get_user_input_data()

    # Choix du test à effectuer
    test_choice = st.sidebar.selectbox("Choose a test", ["T-Test", "Z-Test", "Chi-Square Test", "Linear Regression"])

    if test_choice == "T-Test":
        perform_t_test(data1, data2)
    elif test_choice == "Z-Test":
        perform_z_test(data1, data2)
    elif test_choice == "Chi-Square Test":
        perform_chi_square_test(data1, data2)
    elif test_choice == "Linear Regression":
        perform_linear_regression(data1, data2)

# Exécuter l'application
if __name__ == '__main__':
    perform_tests()
