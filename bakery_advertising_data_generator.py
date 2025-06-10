import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_bakery_advertising_data():
    """
    Génère un dataset réaliste pour une boulangerie locale
    avec des données de campagnes publicitaires sur 24 mois
    """
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    advertising_channels = ['Facebook', 'Google_Ads', 'Radio_Locale', 'Flyers']
    
    # Génération des dates (campagnes hebdomadaires)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=7)
    
    # Initialisation du dataset
    data = []
    
    for date in dates:
        # Facteurs saisonniers qui influencent les performances
        month = date.month
        day_of_week = date.weekday()  # 0=Lundi, 6=Dimanche
        
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
        if month == 12:  # Boost de Noël
            seasonal_factor *= 1.4
        elif month == 1:  # Creux de janvier
            seasonal_factor *= 0.7
            
        weekday_factor = 1.2 if day_of_week in [5, 6] else 1.0
        
        base_budget = 2000
        budget_variation = np.random.normal(0, 200) 
        total_budget = max(1500, base_budget + budget_variation)
        
        time_factor = (date - start_date).days / 365.0
        
        # Évolution des stratégies : plus digital avec le temps
        facebook_share = 0.25 + 0.1 * time_factor + np.random.normal(0, 0.05)
        google_share = 0.30 + 0.15 * time_factor + np.random.normal(0, 0.05)
        radio_share = 0.25 - 0.15 * time_factor + np.random.normal(0, 0.05)
        flyers_share = 0.20 - 0.10 * time_factor + np.random.normal(0, 0.05)
        
        # Normalisation pour que la somme = 1
        total_share = facebook_share + google_share + radio_share + flyers_share
        facebook_share /= total_share
        google_share /= total_share
        radio_share /= total_share
        flyers_share /= total_share
        
        # Calcul des budgets par canal
        budgets = {
            'Facebook': total_budget * facebook_share,
            'Google_Ads': total_budget * google_share,
            'Radio_Locale': total_budget * radio_share,
            'Flyers': total_budget * flyers_share
        }
        
        # Génération des performances par canal
        for channel in advertising_channels:
            budget = budgets[channel]
            
            # Paramètres de performance spécifiques à chaque canal
            if channel == 'Facebook':
               
                cpm = np.random.normal(8, 2)  # Coût pour 1000 impressions
                ctr = np.random.normal(0.02, 0.005)  # Taux de clic
                conversion_rate = np.random.normal(0.03, 0.01)  # Taux de conversion
                avg_order_value = np.random.normal(15, 3)  # Panier moyen
                
            elif channel == 'Google_Ads':
                cpm = np.random.normal(12, 3)
                ctr = np.random.normal(0.035, 0.01)
                conversion_rate = np.random.normal(0.05, 0.015)
                avg_order_value = np.random.normal(18, 4)
                
            elif channel == 'Radio_Locale':
                cpm = np.random.normal(4, 1)
                ctr = np.random.normal(0.001, 0.0005)  # Pas de clic direct
                conversion_rate = np.random.normal(0.02, 0.008)
                avg_order_value = np.random.normal(12, 3)
                
            else:  # Flyers
                cpm = np.random.normal(2, 0.5)
                ctr = np.random.normal(0.005, 0.002)
                conversion_rate = np.random.normal(0.015, 0.008)
                avg_order_value = np.random.normal(10, 2)
        
            impressions = (budget / cpm * 1000) * seasonal_factor * weekday_factor
            impressions = max(0, int(impressions))
            
            clicks = int(impressions * max(0, ctr))
            conversions = int(clicks * max(0, conversion_rate))
            revenue = conversions * max(0, avg_order_value)
        
            impressions += int(np.random.normal(0, impressions * 0.1))
            clicks += int(np.random.normal(0, clicks * 0.15))
            conversions += int(np.random.normal(0, conversions * 0.2))
            revenue += np.random.normal(0, revenue * 0.1)
        
            impressions = max(0, impressions)
            clicks = max(0, clicks)
            conversions = max(0, conversions)
            revenue = max(0, revenue)
        
            cpc = budget / clicks if clicks > 0 else 0
            cpa = budget / conversions if conversions > 0 else 0
            roi = (revenue - budget) / budget if budget > 0 else 0
            
            data.append({
                'Date': date,
                'Canal': channel,
                'Budget': round(budget, 2),
                'Impressions': impressions,
                'Clics': clicks,
                'Conversions': conversions,
                'Chiffre_Affaires': round(revenue, 2),
                'CPC': round(cpc, 2),
                'CPA': round(cpa, 2),
                'ROI': round(roi, 3),
                'Mois': month,
                'Jour_Semaine': day_of_week,
                'Facteur_Saisonnier': round(seasonal_factor, 3),
                'Budget_Total_Semaine': round(total_budget, 2)
            })
    
    return pd.DataFrame(data)

df = generate_bakery_advertising_data()

print("=== APERÇU DU DATASET ===")
print(f"Nombre total de lignes : {len(df)}")
print(f"Période couverte : {df['Date'].min()} à {df['Date'].max()}")
print(f"Canaux publicitaires : {df['Canal'].unique()}")
print("\nPremières lignes du dataset :")
print(df.head(10))

print("\n=== STATISTIQUES PAR CANAL ===")
stats_by_channel = df.groupby('Canal').agg({
    'Budget': ['mean', 'std'],
    'Impressions': ['mean', 'std'],
    'Conversions': ['mean', 'std'],
    'ROI': ['mean', 'std']
}).round(2)
print(stats_by_channel)

print("\n=== ÉVOLUTION TEMPORELLE ===")
monthly_performance = df.groupby(['Mois', 'Canal'])['ROI'].mean().unstack()
print("ROI moyen par mois et par canal :")
print(monthly_performance.round(3))

df.to_csv('bakery_advertising_data.csv', index=False)
print("\n✅ Dataset sauvegardé dans 'bakery_advertising_data.csv'")

print("\n=== QUALITÉ DES DONNÉES ===")
print(f"Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"Données aberrantes (ROI > 2 ou ROI < -0.5) : {len(df[(df['ROI'] > 2) | (df['ROI'] < -0.5)])}")
print(f"Cohérence temporelle : {len(df['Date'].unique())} semaines uniques")