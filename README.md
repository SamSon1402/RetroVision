# RetroVision : Analyse des Étalages Commerciaux

## Résumé Exécutif

RetroVision est une application de vision par ordinateur qui simplifie la surveillance des étalages commerciaux en détectant, classifiant et analysant automatiquement les matériels promotionnels en magasin. Conçue avec une interface distinctive inspirée des jeux rétro pour améliorer l'engagement des utilisateurs, cet outil aide les équipes de merchandising, le personnel marketing terrain et les responsables des opérations commerciales à assurer la conformité promotionnelle et à optimiser les efforts marketing en magasin.

![image](https://github.com/user-attachments/assets/6eb01909-98b9-49eb-a5cc-1ffded0e511a)
![image](https://github.com/user-attachments/assets/1101f61b-9b3d-4221-9de0-847101f239db)
![image](https://github.com/user-attachments/assets/7765c5dd-ffa4-4570-8ad0-b7e4bae75696)


## Valeur Commerciale

### Avantages Principaux

- **Réduction du Temps d'Audit Manuel de 70-80%** : Automatisation de la détection et du comptage des matériels promotionnels qui nécessitaient auparavant une inspection manuelle.

- **Amélioration de la Conformité Promotionnelle** : Identification rapide des matériels promotionnels manquants, mal placés ou endommagés dans plusieurs points de vente.

- **Optimisation du ROI Marketing** : Suivi du placement et de l'efficacité des matériels promotionnels pour prendre des décisions basées sur les données concernant les futures campagnes.

- **Standardisation du Contrôle Qualité** : Mesure objective de la qualité des photos assurant des rapports cohérents entre les équipes terrain.

- **Génération d'Insights Actionnables** : Transformation des images brutes d'étalages en données structurées pouvant orienter les décisions commerciales.

## Capacités Principales

RetroVision combine la vision par ordinateur avancée avec une interface intuitive pour offrir :

1. **Détection Automatisée** : Identifie les bannières, affiches, étiquettes de prix et signalétiques promotionnelles dans les images d'étalages commerciaux.

2. **Classification des Matériaux** : Catégorise les éléments détectés par type de matériau (papier, plastique, carton) pour le suivi des stocks.

3. **Évaluation de la Qualité Photo** : Note les images soumises selon leur netteté, luminosité et contraste pour garantir des données exploitables.

4. **Analytique de Performance** : Compare les résultats de détection entre différents modèles et fournit des métriques d'évaluation.

## Comment Ça Fonctionne

### 1. Capture
Les équipes terrain capturent des images d'étalages à l'aide de smartphones ou tablettes standard.

### 2. Analyse
RetroVision traite les images en utilisant deux modèles d'IA complémentaires :
- **YOLOv8** : Détection rapide et efficace pour traiter de grands lots d'images
- **Mask R-CNN** : Analyse détaillée pour des besoins de précision supérieure

### 3. Rapport
Le système génère des rapports montrant :
- Le décompte de chaque type de matériel promotionnel
- L'analyse de la distribution des matériaux
- La conformité avec les dispositions promotionnelles attendues
- Les métriques de qualité photo pour le feedback aux équipes terrain

## Implémentation Technique

RetroVision est développé comme une application web utilisant :
- **Streamlit** : Pour l'interface utilisateur interactive
- **PyTorch & OpenCV** : Pour les capacités de vision par ordinateur
- **Python** : Pour le traitement backend et l'analyse

L'application présente une esthétique inspirée des jeux rétro avec :
- Des visuels lumineux à fort contraste pour une représentation claire des données
- Des éléments d'interface utilisateur de style pixel pour une expérience distinctive et engageante
- Un retour d'information ludique pour encourager une utilisation appropriée

## Options de Déploiement

1. **Sur Site** : Exécution sur votre réseau d'entreprise pour une sécurité maximale des données
2. **Cloud** : Déploiement sur des services cloud pour un accès de n'importe où
3. **Google Colab** : Environnement de test rapide à des fins d'évaluation

## Mise en Route

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement de RetroVision
streamlit run app.py
```

## Exemple de Cas d'Utilisation

Un détaillant national avec 500 magasins avait besoin de vérifier la conformité promotionnelle dans tous ses points de vente pour une campagne saisonnière majeure. Les méthodes manuelles précédentes nécessitaient :
- 2 heures d'audit par magasin
- 5 membres du personnel dédiés à l'examen des photos
- 2 jours pour compiler et analyser les résultats

Avec RetroVision :
- Temps d'audit réduit à 15 minutes par magasin
- Aucun personnel dédié nécessaire pour l'examen manuel des photos
- Résultats disponibles en temps réel au fur et à mesure du traitement des images
- Augmentation de 60% de la conformité promotionnelle grâce à un feedback plus rapide

## Coordonnées

Pour plus d'informations sur l'implémentation de RetroVision dans vos opérations commerciales, veuillez contacter :

[Vos Coordonnées]
