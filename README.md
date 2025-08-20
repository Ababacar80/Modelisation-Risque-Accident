# 🚗 Modélisation du Risque d'Accident - Assurance Véhicule

Projet de modélisation statistique appliqué à l'analyse du risque d'accident en assurance automobile utilisant la régression logistique.

## 📋 Aperçu du Projet

Ce projet implémente une analyse complète du risque d'accident pour l'assurance véhicule en utilisant des méthodes de régression logistique binaire. Le script principal `Exam_MLG.R` contient une implémentation optimisée pour l'examen de Modèles Linéaires Généralisés (MLG).

## 🗃️ Données Analysées

### Source des Données
- **Fichier principal** : `Base_MLG.csv`
- **Variable cible** : `clm` (accident/pas d'accident)
- **Type de modélisation** : Régression logistique binaire
- **Contexte** : Assurance automobile

### Structure du Dataset

#### Variables Explicatives Identifiées
- **`veh`** : Type de véhicule (SEDAN comme référence)
- **`veh_age`** : Âge du véhicule (4 catégories : ]0,1], ]1,3], ]3,8], ]8,15])
- **`sexe`** : Sexe du conducteur (F/M)
- **`sup`** : Variable de supervision (C comme référence)
- **`age`** : Âge du conducteur (3 comme référence)
- **`value`** : Valeur du véhicule
- **`risque`** : Score de risque

#### Variables Exclues
- **`OBSTAT`** : Supprimée (inutile)
- **`nbclm`** et **`coutclm`** : Exclues (colinéarité parfaite avec `clm`)

## 🏗️ Structure du Projet

```
Modelisation-Risque-Accident/
├── README.md                  # Documentation du projet
├── Exam_MLG.R                # Script principal de régression logistique
├── Base_MLG.csv              # Dataset d'assurance (non visible)
└── [Résultats d'analyse]     # Sorties du modèle
```

## 📊 Méthodologie Implémentée

### 1. **Préparation des Données**
- Suppression des variables problématiques (`OBSTAT`, `nbclm`, `coutclm`)
- Conversion en facteurs avec niveaux de référence appropriés
- Transformations logarithmiques optimales :
  - `value_log = log(value + 1)`
  - `risque_log = log(risque + 0.001)`
- Création de variables dérivées :
  - `value_extreme` : Indicateur de valeur extrême (>95e percentile)
  - `risque_eleve` : Indicateur de risque élevé (>80e percentile)

### 2. **Exploration des Données**
- Distribution de la variable cible `clm`
- Analyse bivariée par type de véhicule
- Moyennes des variables quantitatives par classe
- Matrices de corrélation

### 3. **Modélisation Optimisée**
- **Modèle principal** : Régression logistique avec transformations logarithmiques
- **Sélection de variables** : Procédure stepwise automatique
- **Formule finale** : `clm ~ value_log + risque_log + veh + veh_age + sexe + sup + age`

## 🔧 Analyses Statistiques Réalisées

### Tests de Validité du Modèle

#### **Test Global**
- Statistique de vraisemblance (vs modèle nul)
- Test du chi-deux avec degrés de liberté appropriés

#### **Tests Individuels**
- Tests d'élimination (fonction `drop1`)
- Significativité des coefficients

#### **Tests d'Adéquation**
- **Surdispersion** : Ratio déviance/degrés de liberté
- **Résidus de Pearson** : Test de qualité d'ajustement
- **Test de Hosmer-Lemeshow** : Validation de l'ajustement (g=5 groupes)

### Métriques de Performance

#### **Qualité du Modèle**
- **Pseudo R² (McFadden)** : Pouvoir explicatif
- **AIC/BIC** : Critères d'information pour sélection
- **Déviance résiduelle** : Mesure de l'ajustement

#### **Capacité Prédictive**
- **Courbe ROC** : Performance de classification
- **AUC** : Aire sous la courbe ROC
- **Seuil optimal** : Maximisation de la sensibilité et spécificité
- **Métriques de confusion** :
  - Accuracy (Précision globale)
  - Recall/Sensitivity (Rappel)
  - Precision (Précision positive)
  - F1-Score (Moyenne harmonique)

## 📈 Diagnostic des Résidus

### Analyses Implémentes
- **Résidus normalisés de Pearson** : Détection d'observations atypiques (>2 en valeur absolue)
- **Distance de Cook** : Identification des observations influentes (>1)
- **Graphiques de diagnostic** :
  - Résidus vs Ajustés
  - QQ-plot des résidus
  - Distribution des résidus normalisés
  - Distance de Cook par observation

## 🎯 Interprétation des Résultats

### Odds Ratios
- **Calcul automatique** : `exp(coefficients)` avec intervalles de confiance
- **Présentation élégante** : Utilisation de `broom::tidy()` avec exponentiation
- **Visualisations** : Graphiques des coefficients avec `ggcoef_model()` et `forest_model()`

### Facteurs de Risque Identifiés
Le script identifie automatiquement les **5 facteurs les plus significatifs** :
- Classement par valeur absolue des coefficients
- Distinction entre facteurs augmentant/diminuant le risque
- Calcul des odds ratios correspondants

## 🔍 Validation Prédictive

### Courbe ROC et Optimisation
- **Seuil optimal** : Méthode "best" de la courbe ROC
- **Matrice de confusion** : Utilisation du seuil optimal
- **Comparaison de seuils** : Tests sur différents seuils (0.05, 0.1, optimal, 0.15)

### Prédictions sur Nouveaux Cas
Le script inclut des **profils types** pour validation :
```
Client 1: value=1.5, risque=0.2, véhicule=SEDAN, âge_véhicule=]0,1], sexe=F, sup=C, âge=3
Client 2: value=2.5, risque=0.5, véhicule=COUPE, âge_véhicule=]1,3], sexe=M, sup=C, âge=1  
Client 3: value=3.5, risque=0.8, véhicule=BUS, âge_véhicule=]3,8], sexe=F, sup=D, âge=5
```

## 📊 Visualisations Générées

### Graphiques Automatiques
- **Distribution des probabilités prédites** : Histogramme
- **Boxplot par classe** : Probabilités selon la classe observée
- **Courbe ROC** : Avec indication de l'AUC
- **Graphiques de diagnostic** : 2x2 layout des résidus

## 🏆 Synthèse Finale Automatisée

Le script génère une **synthèse finale complète** incluant :

### **Estimation**
- Méthode utilisée (régression logistique avec transformations)
- Nombre de variables retenues
- AIC final du modèle

### **Significativité**
- Résultat du test global (p < 0.001)
- Nombre de variables significatives sur total

### **Adéquation**
- Pseudo R² (McFadden)
- Résultat du test Hosmer-Lemeshow
- Évaluation de la surdispersion

### **Prévision**
- AUC avec évaluation qualitative (Excellente/Bonne/Correcte)
- Seuil optimal recommandé
- Taux de détection des accidents (Recall)

### **Conclusion Opérationnelle**
- Recommandation de déploiement
- Seuil optimal pour maximiser la détection
- Impact des transformations logarithmiques

## 🔧 Configuration et Dépendances

### Packages R Requis
```r
# Core packages
library(dplyr)          # Manipulation de données
library(ggplot2)        # Visualisations
library(MASS)           # Fonctions statistiques avancées

# Modélisation
library(car)            # Tests de régression
library(ResourceSelection) # Test Hosmer-Lemeshow

# Visualisation des modèles
library(corrplot)       # Matrices de corrélation
library(forestmodel)    # Forest plots
library(effects)        # Graphiques d'effets
library(ggeffects)      # Effets marginaux

# Analyses avancées
library(pROC)           # Courbes ROC
library(ROCR)           # Métriques de performance

# Reporting
library(broom)          # Tidying des modèles
library(gtsummary)      # Tableaux statistiques
```

### Résolution des Conflits
```r
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")
```

## 🎯 Applications Pratiques

### **Tarification d'Assurance**
- Calcul de primes personnalisées basées sur le risque
- Segmentation de la clientèle par profil de risque
- Optimisation du pricing selon les facteurs identifiés

### **Gestion des Risques**
- Identification des profils à haut risque
- Mesures préventives ciblées
- Surveillance continue des portefeuilles

### **Aide à la Décision**
- Acceptation/refus de nouveaux clients
- Ajustement des conditions de couverture
- Allocation des ressources de prévention

## 📝 Points Forts du Modèle

### **Transformations Optimales**
- Amélioration significative par transformations logarithmiques
- Gestion appropriée des variables continues asymétriques

### **Validation Rigoureuse**
- Tests statistiques complets (global, individuel, adéquation)
- Diagnostic des résidus systématique
- Validation croisée par courbe ROC

### **Approche Opérationnelle**
- Seuil optimal calculé automatiquement
- Métriques business (détection d'accidents)
- Recommandations concrètes pour déploiement

## ⚠️ Limitations et Considérations

### **Données**
- Exclusion nécessaire de variables colinéaires
- Dépendance à la qualité des données d'entrée
- Transformations spécifiques au contexte

### **Modélisation**
- Hypothèses de la régression logistique à vérifier
- Stabilité temporelle du modèle à surveiller
- Risque de sur-ajustement avec de petits échantillons

## 🔄 Recommandations d'Amélioration

### **Extensions Possibles**
- Modèles de régression multinomiale (gravité des accidents)
- Intégration de données temporelles/saisonnières
- Techniques d'ensemble (Random Forest, Gradient Boosting)

### **Validation Continue**
- Monitoring des performances en production
- Recalibrage périodique du modèle
- Tests A/B pour validation des seuils

---

**Note** : Ce README est basé sur l'analyse du script `Exam_MLG.R` observé. Il décrit fidèlement les méthodes, tests et analyses implémentés dans le code R pour la modélisation du risque d'accident en assurance véhicule.
