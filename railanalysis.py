#!/usr/bin/env python3
"""
Railway Dataset Analysis & AI Training Examples
=============================================

This script demonstrates how to work with the generated railway dataset
for AI model training and analysis. Includes examples for:

1. Basic data exploration
2. Feature engineering
3. ML model training
4. Performance analysis
5. Visualization examples

Usage: python railway_analysis_examples.py
Prerequisites: pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

class RailwayDatasetAnalyzer:
    def __init__(self):
        """Initialize the analyzer and load all dataset files"""
        print("🚂 Railway Dataset Analyzer")
        print("=" * 50)
        
        # Load all CSV files
        try:
            self.trains = pd.read_csv('train_master_data.csv')
            self.junctions = pd.read_csv('junction_data.csv') 
            self.tracks = pd.read_csv('track_junction_data.csv')
            self.signals = pd.read_csv('signal_data.csv')
            self.schedules = pd.read_csv('schedule_timetable_data.csv')
            self.performance = pd.read_csv('historical_performance_data.csv')
            self.incidents = pd.read_csv('incident_event_data.csv')
            
            # Load rules JSON
            with open('rules_policies_data.json', 'r') as f:
                self.rules = json.load(f)
                
            print("✓ All dataset files loaded successfully")
            self.preprocess_data()
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please run railway_dataset_generator.py first to generate the dataset")
            return
    
    def preprocess_data(self):
        """Basic preprocessing and data type conversions"""
        print("\n📊 Preprocessing data...")
        
        # Convert date/time columns
        date_columns = ['Date', 'Planned_Departure', 'Planned_Arrival', 
                       'Actual_Departure', 'Actual_Arrival']
        
        for col in date_columns:
            if col in self.schedules.columns:
                self.schedules[col] = pd.to_datetime(self.schedules[col])
        
        if 'Date' in self.performance.columns:
            self.performance['Date'] = pd.to_datetime(self.performance['Date'])
            
        if 'Date' in self.incidents.columns:
            self.incidents['Date'] = pd.to_datetime(self.incidents['Date'])
        
        print("✓ Date/time columns converted")
    
    def basic_exploration(self):
        """Perform basic data exploration and statistics"""
        print("\n📈 BASIC DATA EXPLORATION")
        print("=" * 40)
        
        # Dataset overview
        print("Dataset Overview:")
        print(f"  • Trains: {len(self.trains):,}")
        print(f"  • Junctions: {len(self.junctions):,}")
        print(f"  • Tracks: {len(self.tracks):,}")
        print(f"  • Signals: {len(self.signals):,}")
        print(f"  • Schedules: {len(self.schedules):,}")
        print(f"  • Performance Records: {len(self.performance):,}")
        print(f"  • Incidents: {len(self.incidents):,}")
        
        # Train type distribution
        print("\n🚊 Train Type Distribution:")
        train_dist = self.trains['Train_Type'].value_counts()
        for train_type, count in train_dist.items():
            percentage = (count / len(self.trains)) * 100
            print(f"  • {train_type}: {count} ({percentage:.1f}%)")
        
        # Performance statistics
        print("\n⏱️  Performance Statistics:")
        avg_delay = self.performance['Delay_Minutes'].mean()
        on_time_rate = (self.performance['On_Time_Performance'] == 'Yes').mean() * 100
        print(f"  • Average Delay: {avg_delay:.1f} minutes")
        print(f"  • On-Time Performance: {on_time_rate:.1f}%")
        
        # Most common delay reasons
        print("\n🚨 Top Delay Reasons:")
        delay_reasons = self.performance[self.performance['Delay_Minutes'] > 5]['Reason_For_Delay'].value_counts().head(5)
        for reason, count in delay_reasons.items():
            print(f"  • {reason}: {count:,} incidents")
        
        return {
            'total_trains': len(self.trains),
            'avg_delay': avg_delay,
            'on_time_rate': on_time_rate,
            'train_distribution': train_dist.to_dict()
        }
    
    def feature_engineering(self):
        """Create features for ML models"""
        print("\n🔧 FEATURE ENGINEERING")
        print("=" * 30)
        
        # Merge data for ML
        ml_data = self.schedules.merge(self.trains, on='Train_ID')
        ml_data = ml_data.merge(self.performance, on=['Train_ID', 'Date'], how='inner')
        
        # Create time-based features
        ml_data['departure_hour'] = ml_data['Planned_Departure'].dt.hour
        ml_data['departure_day_of_week'] = ml_data['Planned_Departure'].dt.dayofweek
        ml_data['departure_month'] = ml_data['Planned_Departure'].dt.month
        
        # Create categorical features
        ml_data['is_premium'] = ml_data['Train_Type'].isin(['Rajdhani', 'Shatabdi'])
        ml_data['is_freight'] = ml_data['Train_Type'] == 'Freight'
        ml_data['is_rush_hour'] = ml_data['departure_hour'].isin([7,8,9,17,18,19])
        ml_data['is_weekend'] = ml_data['departure_day_of_week'].isin([5,6])
        ml_data['is_monsoon'] = ml_data['departure_month'].isin([6,7,8,9])
        
        # Weather impact
        weather_impact = {'Clear': 0, 'Cloudy': 1, 'Rain': 2, 'Fog': 3, 'Storm': 4, 'Snow': 4}
        ml_data['weather_impact_score'] = ml_data['Weather_Conditions'].map(weather_impact)
        
        # Track usage score
        usage_score = {'Low': 1, 'Medium': 2, 'High': 3}
        ml_data['track_usage_score'] = ml_data['Track_Usage'].map(usage_score)
        
        print(f"✓ Created ML dataset with {len(ml_data):,} records")
        print(f"✓ Generated {len([c for c in ml_data.columns if c.endswith('_score') or c.startswith('is_')])} new features")
        
        self.ml_data = ml_data
        return ml_data
    
    def train_delay_prediction_model(self):
        """Train a model to predict train delays"""
        print("\n🤖 DELAY PREDICTION MODEL")
        print("=" * 35)
        
        if not hasattr(self, 'ml_data'):
            self.feature_engineering()
        
        # Select features and target
        feature_columns = [
            'Max_Speed', 'Priority_Level', 'departure_hour', 'departure_day_of_week',
            'departure_month', 'is_premium', 'is_freight', 'is_rush_hour', 
            'is_weekend', 'is_monsoon', 'weather_impact_score', 'track_usage_score'
        ]
        
        X = self.ml_data[feature_columns]
        y = self.ml_data['Delay_Minutes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"  • Mean Absolute Error: {mae:.2f} minutes")
        print(f"  • R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"  • {row['feature']}: {row['importance']:.3f}")
        
        self.delay_model = model
        return model, mae, r2
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal delay patterns"""
        print("\n📅 SEASONAL PATTERN ANALYSIS")
        print("=" * 40)
        
        # Monthly delay patterns
        monthly_delays = self.performance.groupby(
            self.performance['Date'].dt.month
        )['Delay_Minutes'].agg(['mean', 'count']).reset_index()
        
        monthly_delays['month_name'] = pd.to_datetime(monthly_delays['Date'], format='%m').dt.strftime('%B')
        
        print("📊 Average Delays by Month:")
        for _, row in monthly_delays.iterrows():
            print(f"  • {row['month_name']}: {row['mean']:.1f} min ({row['count']:,} records)")
        
        # Weather impact analysis
        weather_impact = self.performance.groupby('Weather_Conditions')['Delay_Minutes'].agg([
            'mean', 'median', 'count'
        ]).round(2)
        
        print("\n🌤️  Weather Impact on Delays:")
        for weather, stats in weather_impact.iterrows():
            print(f"  • {weather}: Avg {stats['mean']:.1f} min, Median {stats['median']:.1f} min ({stats['count']:,} records)")
        
        return monthly_delays, weather_impact
    
    def incident_analysis(self):
        """Analyze incident patterns and impacts"""
        print("\n🚨 INCIDENT ANALYSIS")
        print("=" * 25)
        
        # Incident type frequency and impact
        incident_stats = self.incidents.groupby('Event_Type').agg({
            'Impact_Minutes': ['mean', 'median', 'sum'],
            'Event_ID': 'count',
            'Cost_Impact': 'mean'
        }).round(2)
        
        incident_stats.columns = ['Avg_Impact', 'Median_Impact', 'Total_Impact', 'Frequency', 'Avg_Cost']
        incident_stats = incident_stats.sort_values('Total_Impact', ascending=False)
        
        print("📊 Incident Types by Total Impact:")
        for event_type, stats in incident_stats.iterrows():
            print(f"  • {event_type}:")
            print(f"    - Frequency: {stats['Frequency']:,} incidents")
            print(f"    - Avg Impact: {stats['Avg_Impact']:.1f} minutes")
            print(f"    - Total Impact: {stats['Total_Impact']:,.0f} minutes")
            print(f"    - Avg Cost: ₹{stats['Avg_Cost']:,.0f}")
        
        # Monthly incident patterns
        monthly_incidents = self.incidents.groupby(
            self.incidents['Date'].dt.month
        ).size().reset_index()
        monthly_incidents['month_name'] = pd.to_datetime(monthly_incidents['Date'], format='%m').dt.strftime('%B')
        
        print(f"\n📅 Incidents by Month:")
        for _, row in monthly_incidents.iterrows():
            print(f"  • {row['month_name']}: {row[0]:,} incidents")
        
        return incident_stats
    
    def route_performance_analysis(self):
        """Analyze performance by route characteristics"""
        print("\n🛤️  ROUTE PERFORMANCE ANALYSIS")
        print("=" * 40)
        
        # Parse route data (from JSON strings)
        route_lengths = []
        for route_str in self.schedules['Route'].dropna():
            try:
                route = json.loads(route_str) if isinstance(route_str, str) else route_str
                route_lengths.append(len(route))
            except:
                route_lengths.append(0)
        
        route_perf = self.schedules.copy()
        route_perf['route_length'] = route_lengths
        
        # Merge with performance data
        route_analysis = route_perf.merge(
            self.performance[['Train_ID', 'Date', 'Delay_Minutes']], 
            on=['Train_ID', 'Date'], 
            how='inner'
        )
        
        # Group by route length
        route_stats = route_analysis.groupby('route_length')['Delay_Minutes'].agg([
            'mean', 'median', 'count'
        ]).round(2)
        
        print("📊 Delay by Route Length (number of stations):")
        for route_len, stats in route_stats.iterrows():
            if stats['count'] > 100:  # Only show routes with sufficient data
                print(f"  • {route_len} stations: Avg {stats['mean']:.1f} min, Median {stats['median']:.1f} min ({stats['count']:,} journeys)")
        
        return route_stats
    
    def generate_operational_insights(self):
        """Generate actionable operational insights"""
        print("\n💡 OPERATIONAL INSIGHTS")
        print("=" * 35)
        
        insights = []
        
        # Train type performance
        train_performance = self.ml_data.groupby('Train_Type')['Delay_Minutes'].agg(['mean', 'count'])
        best_performing = train_performance['mean'].idxmin()
        worst_performing = train_performance['mean'].idxmax()
        
        insights.append(f"🏆 Best performing train type: {best_performing} (avg delay: {train_performance.loc[best_performing, 'mean']:.1f} min)")
        insights.append(f"📉 Needs improvement: {worst_performing} (avg delay: {train_performance.loc[worst_performing, 'mean']:.1f} min)")
        
        # Rush hour impact
        rush_hour_delay = self.ml_data[self.ml_data['is_rush_hour']]['Delay_Minutes'].mean()
        non_rush_delay = self.ml_data[~self.ml_data['is_rush_hour']]['Delay_Minutes'].mean()
        rush_impact = rush_hour_delay - non_rush_delay
        
        if rush_impact > 0:
            insights.append(f"🚦 Rush hour impact: +{rush_impact:.1f} min average delay during peak hours")
        
        # Weather impact
        storm_delay = self.performance[self.performance['Weather_Conditions'] == 'Storm']['Delay_Minutes'].mean()
        clear_delay = self.performance[self.performance['Weather_Conditions'] == 'Clear']['Delay_Minutes'].mean()
        
        insights.append(f"🌩️  Storm weather impact: +{(storm_delay - clear_delay):.1f} min vs clear weather")
        
        # High-impact incidents
        critical_incidents = self.incidents[self.incidents['Impact_Minutes'] > 60]
        insights.append(f"⚠️  Critical incidents (>1hr impact): {len(critical_incidents):,} events causing {critical_incidents['Impact_Minutes'].sum():,.0f} total delay minutes")
        
        # Track utilization
        high_usage_delays = self.performance[self.performance['Track_Usage'] == 'High']['Delay_Minutes'].mean()
        low_usage_delays = self.performance[self.performance['Track_Usage'] == 'Low']['Delay_Minutes'].mean()
        
        insights.append(f"🚧 High track utilization impact: +{(high_usage_delays - low_usage_delays):.1f} min vs low utilization")
        
        print("\n📋 Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        return insights
    
    def create_visualizations(self):
        """Generate key visualizations (requires matplotlib/seaborn)"""
        print("\n📊 GENERATING VISUALIZATIONS")
        print("=" * 40)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Railway Dataset Analysis Dashboard', fontsize=16)
            
            # 1. Train type distribution
            self.trains['Train_Type'].value_counts().plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Train Type Distribution')
            axes[0,0].set_xlabel('Train Type')
            axes[0,0].set_ylabel('Count')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Monthly delay patterns
            monthly_delays = self.performance.groupby(self.performance['Date'].dt.month)['Delay_Minutes'].mean()
            monthly_delays.plot(kind='line', ax=axes[0,1], marker='o')
            axes[0,1].set_title('Average Delays by Month')
            axes[0,1].set_xlabel('Month')
            axes[0,1].set_ylabel('Average Delay (minutes)')
            
            # 3. Weather impact on delays
            weather_delays = self.performance.groupby('Weather_Conditions')['Delay_Minutes'].mean().sort_values(ascending=False)
            weather_delays.plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Weather Impact on Delays')
            axes[1,0].set_xlabel('Weather Condition')
            axes[1,0].set_ylabel('Average Delay (minutes)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Incident frequency by type
            incident_freq = self.incidents['Event_Type'].value_counts()
            incident_freq.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
            axes[1,1].set_title('Incident Types Distribution')
            axes[1,1].set_ylabel('')
            
            plt.tight_layout()
            plt.savefig('railway_analysis_dashboard.png', dpi=300, bbox_inches='tight')
            print("✓ Saved railway_analysis_dashboard.png")
            
            # Additional correlation heatmap
            if hasattr(self, 'ml_data'):
                plt.figure(figsize=(10, 8))
                numeric_cols = ['Max_Speed', 'Priority_Level', 'Delay_Minutes', 
                               'weather_impact_score', 'track_usage_score']
                correlation_matrix = self.ml_data[numeric_cols].corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Heatmap')
                plt.tight_layout()
                plt.savefig('railway_correlation_heatmap.png', dpi=300, bbox_inches='tight')
                print("✓ Saved railway_correlation_heatmap.png")
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not available - skipping visualizations")
            print("   Install with: pip install matplotlib seaborn")
    
    def export_ml_ready_dataset(self):
        """Export cleaned, ML-ready dataset"""
        print("\n💾 EXPORTING ML-READY DATASET")
        print("=" * 40)
        
        if not hasattr(self, 'ml_data'):
            self.feature_engineering()
        
        # Select relevant columns for ML
        ml_columns = [
            'Train_ID', 'Date', 'Train_Type', 'Max_Speed', 'Priority_Level',
            'Weather_Conditions', 'Track_Usage', 'Delay_Minutes',
            'departure_hour', 'departure_day_of_week', 'departure_month',
            'is_premium', 'is_freight', 'is_rush_hour', 'is_weekend', 
            'is_monsoon', 'weather_impact_score', 'track_usage_score'
        ]
        
        ml_dataset = self.ml_data[ml_columns].copy()
        
        # Export to CSV
        ml_dataset.to_csv('railway_ml_ready_dataset.csv', index=False)
        print(f"✓ Exported ML-ready dataset: railway_ml_ready_dataset.csv")
        print(f"  • Shape: {ml_dataset.shape}")
        print(f"  • Features: {len(ml_columns)} columns")
        print(f"  • Target: Delay_Minutes")
        
        return ml_dataset
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚂 RAILWAY DATASET COMPLETE ANALYSIS")
        print("=" * 50)
        
        # Run all analysis modules
        basic_stats = self.basic_exploration()
        ml_data = self.feature_engineering()
        model_results = self.train_delay_prediction_model()
        seasonal_analysis = self.analyze_seasonal_patterns()
        incident_stats = self.incident_analysis()
        route_stats = self.route_performance_analysis()
        insights = self.generate_operational_insights()
        self.create_visualizations()
        ml_dataset = self.export_ml_ready_dataset()
        
        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETE!")
        print("Files generated:")
        print("  • railway_analysis_dashboard.png")
        print("  • railway_correlation_heatmap.png")
        print("  • railway_ml_ready_dataset.csv")
        print("\nDataset is ready for AI training and further analysis!")
        
        return {
            'basic_stats': basic_stats,
            'model_performance': model_results,
            'insights': insights,
            'ml_dataset': ml_dataset
        }

# Example usage and standalone execution
def demo_usage():
    """Demonstrate key functionality with sample code"""
    print("\n" + "="*60)
    print("DEMO: Sample Analysis Code")
    print("="*60)
    
    sample_code = '''
# Load and analyze railway dataset
analyzer = RailwayDatasetAnalyzer()

# Quick statistics
stats = analyzer.basic_exploration()
print(f"On-time performance: {stats['on_time_rate']:.1f}%")

# Train ML model
model, mae, r2 = analyzer.train_delay_prediction_model()
print(f"Model accuracy: {r2:.3f} R²")

# Generate insights
insights = analyzer.generate_operational_insights()

# Export ML-ready data
ml_data = analyzer.export_ml_ready_dataset()
print(f"ML dataset shape: {ml_data.shape}")
'''
    
    print(sample_code)

if __name__ == "__main__":
    # Check if dataset files exist
    required_files = [
        'train_master_data.csv',
        'historical_performance_data.csv',
        'schedule_timetable_data.csv'
    ]
    
    import os
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing dataset files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n🔧 Please run railway_dataset_generator.py first!")
        demo_usage()
    else:
        # Run complete analysis
        analyzer = RailwayDatasetAnalyzer()
        results = analyzer.run_complete_analysis()