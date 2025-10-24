# =====================================================
# 1. INITIALISATION
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# a. Afficher la version de pandas
print("Version de pandas :", pd.__version__)

# =====================================================
# 2. CHARGEMENT DES DONNÉES
# =====================================================

# a. Charger le fichier Titanic (assure-toi que titanic.csv est dans le même dossier)
data = pd.read_csv("titanic.csv")

# =====================================================
# 3. INSPECTION INITIALE
# =====================================================

print("\n--- Aperçu des 5 premières lignes ---")
print(data.head())

print("\n--- Aperçu des 5 dernières lignes ---")
print(data.tail())

# (Optionnel) Améliorer l’affichage de colonnes larges
pd.set_option('display.max_columns', None)

# =====================================================
# 4. STRUCTURE ET TYPES
# =====================================================

print("\n--- Informations générales ---")
data.info()

print("\n--- Types de données ---")
print(data.dtypes)

print("\n--- Statistiques descriptives (numériques) ---")
print(data.describe())

print("\n--- Statistiques descriptives (toutes colonnes) ---")
print(data.describe(include='all'))

# =====================================================
# 5. EXPLORATION DE COLONNES INDIVIDUELLES
# =====================================================

# a. Accéder à une colonne de deux manières
print("\n--- Accès à la colonne Age ---")
print(data.Age.head())
print(data['Age'].head())

# b. value_counts() sur des colonnes catégorielles
print("\n--- Répartition par sexe ---")
print(data['Sex'].value_counts())

print("\n--- Répartition par port d’embarquement ---")
print(data['Embarked'].value_counts())

print("\n--- Répartition par classe ---")
print(data['Pclass'].value_counts())

# =====================================================
# 6. VALEURS UNIQUES
# =====================================================

print("\n--- Nombre de valeurs uniques par colonne ---")
print(data.nunique())

# =====================================================
# 7. CONVERSION DE TYPES
# =====================================================

# Exemple : convertir 'Pclass' en catégorie
data['Pclass'] = data['Pclass'].astype('category')
print("\nType de Pclass après conversion :", data['Pclass'].dtype)

# =====================================================
# 8. FILTRAGE
# =====================================================

print("\n--- Passagers embarqués à Cherbourg ---")
print(data[data['Embarked'] == 'C'][['Name', 'Embarked']].head())

print("\n--- Passagers jeunes et tarif raisonnable (Age < 30 & Fare < 100) ---")
print(data[(data['Age'] < 30) & (data['Fare'] < 100)][['Name', 'Age', 'Fare']].head())

print("\n--- Passagers très riches OU très âgés (Fare > 500 | Age > 70) ---")
print(data[(data['Fare'] > 500) | (data['Age'] > 70)][['Name', 'Age', 'Fare']])

# =====================================================
# 9. DÉTECTION DES VALEURS MANQUANTES
# =====================================================

print("\n--- Valeurs manquantes par colonne ---")
print(data.isnull().sum())

# =====================================================
# 10. GESTION DES VALEURS MANQUANTES
# =====================================================

# a. dropna()
data_dropna = data.dropna()
print("\nTaille après dropna :", data_dropna.shape)

# b. fillna() (exemple avec moyenne ou mode)
data_fillna = data.copy()
data_fillna['Age'] = data_fillna['Age'].fillna(data_fillna['Age'].median())
data_fillna['Embarked'] = data_fillna['Embarked'].fillna(data_fillna['Embarked'].mode()[0])

print("\nTaille après fillna :", data_fillna.shape)

# =====================================================
# 11. NETTOYAGE / SUPPRESSION DE COLONNES
# =====================================================

colonnes_a_supprimer = ['Name', 'Ticket', 'Cabin']
data_clean = data_fillna.drop(columns=colonnes_a_supprimer)

print("\nNouvelle forme du dataset :", data_clean.shape)

# =====================================================
# 12. VÉRIFICATIONS FINALES
# =====================================================

print("\n--- Aperçu final du dataset nettoyé ---")
print(data_clean.head())

print("\n--- Vérification des valeurs manquantes ---")
print(data_clean.isnull().sum())

# =====================================================
# BONUS : VISUALISATIONS SIMPLES
# =====================================================

# Taux de survie par sexe
sns.countplot(x='Sex', hue='Survived', data=data_clean)
plt.title('Taux de survie par sexe')
plt.show()

# Taux de survie par classe
sns.countplot(x='Pclass', hue='Survived', data=data_clean)
plt.title('Taux de survie par classe')
plt.show()
