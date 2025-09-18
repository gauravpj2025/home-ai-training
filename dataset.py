# !/usr/bin/env python3
"""
Railway Traffic Control AI Dataset Generator
============================================

This script generates a comprehensive synthetic dataset for railway traffic control AI simulation.
Includes 7 interconnected tables with realistic data patterns and relationships.

Dataset includes:
1. Train Master Data (2,000+ records)
2. Track & Junction Data (1,000+ junctions, 3,000+ tracks)  
3. Signal Data (5,000+ signals)
4. Schedule & Timetable Data (50,000+ daily entries)
5. Historical Performance Data (500,000+ records)
6. Incident/Event Data (50,000+ disruptions)
7. Rules/Policies Data (AI reference rules)

Usage: python railway_dataset_generator.py
Output: Multiple CSV files for each table
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv
import json
import os

class RailwayDatasetGenerator:
    def __init__(self):
        # Set seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Define realistic parameters
        self.train_types = ['Passenger', 'Express', 'Freight', 'Local', 'Superfast', 'Shatabdi', 'Rajdhani', 'Metro']
        self.weather_conditions = ['Clear', 'Rain', 'Fog', 'Storm', 'Snow', 'Cloudy', 'Windy']
        self.track_types = ['Single Line', 'Double Line', 'Loop', 'Junction Track', 'Yard Track']
        self.signal_states = ['Red', 'Yellow', 'Green']
        self.delay_reasons = ['Signal failure', 'Congestion', 'Weather', 'Maintenance', 'Technical fault', 'Track work', 'Late arrival', 'Crew delay', 'None']
        self.event_types = ['Signal failure', 'Accident', 'Emergency train', 'Track maintenance', 'Power failure', 'Equipment failure', 'Medical emergency']
        self.congestion_levels = ['Low', 'Medium', 'High']
        
        # Indian railway stations for realism
        self.major_stations = [
            'New Delhi', 'Mumbai Central', 'Kolkata', 'Chennai Central', 'Bangalore', 'Hyderabad', 'Pune', 'Ahmedabad',
            'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Bhopal', 'Jaipur', 'Surat', 'Vadodara', 'Agra', 'Nashik',
            'Varanasi', 'Patna', 'Gwalior', 'Vijayawada', 'Coimbatore', 'Madurai', 'Trivandrum', 'Kochi', 'Mysore',
            'Mangalore', 'Hubli', 'Jodhpur', 'Bikaner', 'Udaipur', 'Kota', 'Ajmer', 'Allahabad', 'Gorakhpur',
            'Bareilly', 'Meerut', 'Dehradun', 'Haridwar', 'Amritsar', 'Jalandhar', 'Ludhiana', 'Chandigarh',
            'Shimla', 'Jammu', 'Srinagar', 'Guwahati', 'Dibrugarh', 'Silchar', 'Bhubaneswar', 'Cuttack',
            'Puri', 'Rourkela', 'Durgapur', 'Asansol', 'Siliguri', 'Jalpaiguri', 'Ranchi', 'Dhanbad',
            'Bokaro', 'Jamshedpur', 'Raipur', 'Bilaspur', 'Bhilai', 'Korba', 'Jabalpur', 'Ujjain',
            'Ratlam', 'Khandwa', 'Itarsi', 'Bina', 'Katni', 'Satna', 'Rewa', 'Singrauli', 'Aurangabad',
            'Solapur', 'Sangli', 'Kolhapur', 'Belgaum', 'Davangere', 'Bellary', 'Raichur', 'Gulbarga'
        ]
        
        # Generate extended station network
        self.stations = self._generate_station_network()
        
        # Color codes for train visualization
        self.color_codes = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', 
                           '#800080', '#008000', '#000080', '#808000', '#008080', '#C0C0C0', '#808080']
        
        print(f"Initialized Railway Dataset Generator with {len(self.stations)} stations")
    
    def _generate_station_network(self):
        """Generate extended station network by adding suffixes to major stations"""
        stations = self.major_stations.copy()
        suffixes = ['Junction', 'Central', 'Terminal', 'City', 'Cantt', 'Road', 'Town', 'Nagar', 'Colony', 'Gate']
        
        # Add variations of major stations
        for station in self.major_stations[:50]:  # Use first 50 as base
            for suffix in suffixes[:3]:  # Use first 3 suffixes per station
                if len(stations) < 1200:
                    new_name = f"{station} {suffix}"
                    if new_name not in stations:
                        stations.append(new_name)
        
        return stations[:1200]  # Return first 1200 unique stations
    
    def generate_train_master_data(self, n_trains=2000):
        """Generate Train Master Data table"""
        print(f"Generating Train Master Data ({n_trains} trains)...")
        
        train_data = []
        for i in range(n_trains):
            train_type = np.random.choice(self.train_types)
            
            # Realistic parameters based on train type
            if train_type == 'Freight':
                max_speed = np.random.randint(60, 90)
                avg_speed = max_speed - np.random.randint(10, 25)
                capacity = np.random.randint(1000, 5000)  # tons
                priority = 3  # lowest priority
                length = np.random.randint(200, 500)  # longer freight trains
            elif train_type in ['Shatabdi', 'Rajdhani']:
                max_speed = np.random.randint(140, 180)
                avg_speed = max_speed - np.random.randint(15, 30)
                capacity = np.random.randint(400, 800)  # seats
                priority = 1  # highest priority
                length = np.random.randint(200, 350)
            elif train_type == 'Express':
                max_speed = np.random.randint(100, 130)
                avg_speed = max_speed - np.random.randint(15, 25)
                capacity = np.random.randint(800, 1500)
                priority = 1 if np.random.random() > 0.7 else 2  # Some express trains have high priority
                length = np.random.randint(250, 400)
            elif train_type == 'Metro':
                max_speed = np.random.randint(70, 100)
                avg_speed = max_speed - np.random.randint(10, 20)
                capacity = np.random.randint(800, 1200)
                priority = 2
                length = np.random.randint(100, 200)
            else:  # Passenger, Local
                max_speed = np.random.randint(80, 110)
                avg_speed = max_speed - np.random.randint(10, 20)
                capacity = np.random.randint(600, 1200)
                priority = 2
                length = np.random.randint(150, 300)
            
            train_data.append({
                'Train_ID': f"TR{str(i+1).zfill(4)}",
                'Train_Name': f"{np.random.choice(self.stations).replace(' ', '_')}_{train_type}_{i+1}",
                'Train_Type': train_type,
                'Max_Speed': max_speed,
                'Avg_Speed': avg_speed,
                'Train_Length': length,
                'Capacity': capacity,
                'Priority_Level': priority,
                'Color_Code': np.random.choice(self.color_codes)
            })
        
        df = pd.DataFrame(train_data)
        df.to_csv('train_master_data.csv', index=False)
        print(f"✓ Created train_master_data.csv with {len(df)} records")
        return df
    
    def generate_track_junction_data(self, n_junctions=1000, n_tracks=3000):
        """Generate Track & Junction Data"""
        print(f"Generating Track & Junction Data ({n_junctions} junctions, {n_tracks} tracks)...")
        
        # First create junctions
        junction_data = []
        for i in range(n_junctions):
            # Select station as junction location
            station = np.random.choice(self.stations)
            
            junction_data.append({
                'Junction_ID': f"JN{str(i+1).zfill(4)}",
                'Junction_Name': station,
                'Latitude': np.random.uniform(8.0, 37.0),  # India's latitude range
                'Longitude': np.random.uniform(68.0, 97.0),  # India's longitude range
                'Junction_Type': np.random.choice(['Major', 'Minor', 'Terminal', 'Yard']),
                'Platform_Count': np.random.randint(2, 12),
                'Max_Capacity': np.random.randint(5, 50)  # trains at a time
            })
        
        junction_df = pd.DataFrame(junction_data)
        junction_df.to_csv('junction_data.csv', index=False)
        
        # Now create tracks connecting junctions
        track_data = []
        junction_ids = [f"JN{str(i+1).zfill(4)}" for i in range(n_junctions)]
        
        for i in range(n_tracks):
            start_junction = np.random.choice(junction_ids)
            end_junction = np.random.choice(junction_ids)
            
            # Ensure start and end are different
            while end_junction == start_junction:
                end_junction = np.random.choice(junction_ids)
            
            track_type = np.random.choice(self.track_types)
            length = np.random.uniform(5.0, 500.0)  # km
            
            # Capacity based on track type
            if track_type == 'Double Line':
                max_capacity = np.random.randint(4, 10)
            elif track_type == 'Single Line':
                max_capacity = np.random.randint(1, 3)
            else:
                max_capacity = np.random.randint(2, 6)
            
            # Generate signal and halt positions as coordinate lists
            n_signals = max(1, int(length // 20))  # Signal every ~20km
            n_halts = max(0, int(length // 50))    # Station every ~50km
            
            signal_positions = [round(np.random.uniform(0, length), 2) for _ in range(n_signals)]
            halt_positions = [round(np.random.uniform(0, length), 2) for _ in range(n_halts)]
            
            track_data.append({
                'Track_ID': f"TK{str(i+1).zfill(4)}",
                'Start_Junction': start_junction,
                'End_Junction': end_junction,
                'Track_Length': round(length, 2),
                'Max_Capacity': max_capacity,
                'Signal_Positions': json.dumps(signal_positions),
                'Halt_Positions': json.dumps(halt_positions),
                'Track_Type': track_type,
                'Electrified': np.random.choice(['Yes', 'No'], p=[0.7, 0.3]),
                'Max_Speed_Limit': np.random.choice([80, 100, 120, 140, 160])
            })
        
        track_df = pd.DataFrame(track_data)
        track_df.to_csv('track_junction_data.csv', index=False)
        print(f"✓ Created junction_data.csv with {len(junction_df)} records")
        print(f"✓ Created track_junction_data.csv with {len(track_df)} records")
        return junction_df, track_df
    
    def generate_signal_data(self, track_df, n_signals=5000):
        """Generate Signal Data based on tracks"""
        print(f"Generating Signal Data ({n_signals} signals)...")
        
        signal_data = []
        track_ids = track_df['Track_ID'].tolist()
        
        for i in range(n_signals):
            # Select random track for signal location
            associated_track = np.random.choice(track_ids)
            track_info = track_df[track_df['Track_ID'] == associated_track].iloc[0]
            
            # Position along the track
            position_km = np.random.uniform(0, track_info['Track_Length'])
            
            # Signal cycle times (realistic timing)
            red_time = np.random.randint(60, 180)    # 1-3 minutes
            yellow_time = np.random.randint(5, 15)    # 5-15 seconds
            green_time = np.random.randint(90, 300)   # 1.5-5 minutes
            
            # Determine controlled tracks (this signal plus nearby tracks)
            controlled_tracks = [associated_track]
            if np.random.random() > 0.7:  # 30% chance of controlling multiple tracks
                additional_tracks = np.random.choice(track_ids, size=np.random.randint(1, 3), replace=False)
                controlled_tracks.extend(additional_tracks.tolist())
            
            signal_data.append({
                'Signal_ID': f"SG{str(i+1).zfill(4)}",
                'Location': f"{track_info['Start_Junction']}-{track_info['End_Junction']}",
                'Track_ID': associated_track,
                'Position_KM': round(position_km, 2),
                'Initial_State': np.random.choice(self.signal_states),
                'Red_Time_Sec': red_time,
                'Yellow_Time_Sec': yellow_time,
                'Green_Time_Sec': green_time,
                'Controlled_Tracks': json.dumps(controlled_tracks),
                'Signal_Type': np.random.choice(['Automatic', 'Manual', 'Semi-Automatic']),
                'Last_Maintenance': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        
        df = pd.DataFrame(signal_data)
        df.to_csv('signal_data.csv', index=False)
        print(f"✓ Created signal_data.csv with {len(df)} records")
        return df
    
    def generate_schedule_data(self, train_df, junction_df, n_schedules=50000):
        """Generate Schedule & Timetable Data"""
        print(f"Generating Schedule & Timetable Data ({n_schedules} entries)...")
        
        schedule_data = []
        train_ids = train_df['Train_ID'].tolist()
        junction_names = junction_df['Junction_Name'].tolist()
        
        # Generate schedules over 365 days
        base_date = datetime(2024, 1, 1)
        
        for i in range(n_schedules):
            train_id = np.random.choice(train_ids)
            train_info = train_df[train_df['Train_ID'] == train_id].iloc[0]
            
            # Generate route (3-8 junctions)
            route_length = np.random.randint(3, 8)
            route = np.random.choice(junction_names, size=route_length, replace=False).tolist()
            
            # Random date within year
            schedule_date = base_date + timedelta(days=np.random.randint(0, 365))
            
            # Departure time (spread throughout day, with peaks during rush hours)
            hour_weights = [0.02, 0.01, 0.01, 0.01, 0.02, 0.08, 0.12, 0.08, 0.06, 0.04, 0.04, 0.04,
                           0.04, 0.04, 0.04, 0.04, 0.06, 0.08, 0.12, 0.08, 0.04, 0.02, 0.02, 0.01]
            departure_hour = np.random.choice(range(24), p=hour_weights)
            departure_minute = np.random.randint(0, 60)
            
            planned_departure = schedule_date.replace(hour=departure_hour, minute=departure_minute)
            
            # Journey time based on train type and route length
            if train_info['Train_Type'] in ['Shatabdi', 'Rajdhani']:
                journey_hours = route_length * np.random.uniform(0.8, 1.2)
            elif train_info['Train_Type'] == 'Express':
                journey_hours = route_length * np.random.uniform(1.2, 1.8)
            elif train_info['Train_Type'] == 'Freight':
                journey_hours = route_length * np.random.uniform(2.0, 3.5)
            else:
                journey_hours = route_length * np.random.uniform(1.5, 2.5)
            
            planned_arrival = planned_departure + timedelta(hours=journey_hours)
            
            # Add realistic delays
            delay_probability = 0.3  # 30% chance of delay
            if np.random.random() < delay_probability:
                delay_minutes = np.random.exponential(15)  # Exponential distribution for delays
                delay_minutes = min(delay_minutes, 180)  # Cap at 3 hours
                
                actual_departure = planned_departure + timedelta(minutes=delay_minutes)
                actual_arrival = planned_arrival + timedelta(minutes=delay_minutes)
            else:
                actual_departure = planned_departure
                actual_arrival = planned_arrival
            
            schedule_data.append({
                'Schedule_ID': f"SC{str(i+1).zfill(6)}",
                'Train_ID': train_id,
                'Date': schedule_date.date(),
                'Planned_Departure': planned_departure,
                'Planned_Arrival': planned_arrival,
                'Actual_Departure': actual_departure,
                'Actual_Arrival': actual_arrival,
                'Route': json.dumps(route),
                'Distance_KM': round(route_length * np.random.uniform(50, 200), 1),
                'Passenger_Load': np.random.randint(int(train_info['Capacity'] * 0.3), 
                                                   int(train_info['Capacity'] * 1.1))
            })
        
        df = pd.DataFrame(schedule_data)
        df.to_csv('schedule_timetable_data.csv', index=False)
        print(f"✓ Created schedule_timetable_data.csv with {len(df)} records")
        return df
    
    def generate_historical_performance_data(self, train_df, schedule_df, n_records=500000):
        """Generate Historical Performance Data"""
        print(f"Generating Historical Performance Data ({n_records} records)...")
        
        performance_data = []
        train_ids = train_df['Train_ID'].tolist()
        
        # Generate data over 2 years
        base_date = datetime(2022, 1, 1)
        
        for i in range(n_records):
            train_id = np.random.choice(train_ids)
            
            # Random date over 2 years
            performance_date = base_date + timedelta(days=np.random.randint(0, 730))
            
            # Weather bias - more delays in bad weather
            weather = np.random.choice(self.weather_conditions, 
                                     p=[0.4, 0.15, 0.1, 0.05, 0.05, 0.2, 0.05])
            
            # Delay calculation based on weather and other factors
            if weather in ['Storm', 'Fog', 'Snow']:
                delay_minutes = np.random.exponential(25)
                reason = np.random.choice(self.delay_reasons, 
                                        p=[0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])
            elif weather == 'Rain':
                delay_minutes = np.random.exponential(12)
                reason = np.random.choice(self.delay_reasons,
                                        p=[0.15, 0.25, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])
            else:
                delay_minutes = np.random.exponential(8)
                reason = np.random.choice(self.delay_reasons,
                                        p=[0.2, 0.3, 0.05, 0.15, 0.15, 0.05, 0.05, 0.02, 0.03])
            
            delay_minutes = min(delay_minutes, 240)  # Cap at 4 hours
            
            # Track usage correlation with delays
            if delay_minutes > 30:
                track_usage = np.random.choice(self.congestion_levels, p=[0.1, 0.3, 0.6])
            elif delay_minutes > 10:
                track_usage = np.random.choice(self.congestion_levels, p=[0.2, 0.5, 0.3])
            else:
                track_usage = np.random.choice(self.congestion_levels, p=[0.6, 0.3, 0.1])
            
            # Wait time at junctions
            if track_usage == 'High':
                avg_wait_time = np.random.uniform(5, 25)
            elif track_usage == 'Medium':
                avg_wait_time = np.random.uniform(2, 10)
            else:
                avg_wait_time = np.random.uniform(0, 5)
            
            performance_data.append({
                'Performance_ID': f"PF{str(i+1).zfill(7)}",
                'Train_ID': train_id,
                'Date': performance_date.date(),
                'Weather_Conditions': weather,
                'Delay_Minutes': round(delay_minutes, 1),
                'Reason_For_Delay': reason,
                'Track_Usage': track_usage,
                'Avg_Wait_Time_Minutes': round(avg_wait_time, 1),
                'Fuel_Consumption': round(np.random.uniform(500, 2000), 2),  # Liters
                'On_Time_Performance': 'Yes' if delay_minutes <= 5 else 'No',
                'Route_Efficiency': round(np.random.uniform(0.7, 1.0), 3)
            })
        
        df = pd.DataFrame(performance_data)
        df.to_csv('historical_performance_data.csv', index=False)
        print(f"✓ Created historical_performance_data.csv with {len(df)} records")
        return df
    
    def generate_incident_event_data(self, train_df, n_events=50000):
        """Generate Incident/Event Data for AI learning"""
        print(f"Generating Incident/Event Data ({n_events} events)...")
        
        event_data = []
        train_ids = train_df['Train_ID'].tolist()
        
        # Generate events over 3 years
        base_date = datetime(2021, 1, 1)
        
        for i in range(n_events):
            train_id = np.random.choice(train_ids) if np.random.random() > 0.1 else None  # 10% system-wide events
            
            # Random date over 3 years
            event_date = base_date + timedelta(days=np.random.randint(0, 1095))
            
            event_type = np.random.choice(self.event_types,
                                        p=[0.25, 0.05, 0.15, 0.20, 0.15, 0.15, 0.05])
            
            # Impact based on event type
            if event_type == 'Accident':
                impact_minutes = np.random.uniform(60, 480)  # 1-8 hours
                resolved_minutes = np.random.uniform(120, 720)  # 2-12 hours
            elif event_type == 'Signal failure':
                impact_minutes = np.random.uniform(15, 120)  # 15 minutes - 2 hours
                resolved_minutes = np.random.uniform(30, 180)  # 30 minutes - 3 hours
            elif event_type == 'Track maintenance':
                impact_minutes = np.random.uniform(30, 180)  # 30 minutes - 3 hours
                resolved_minutes = impact_minutes  # Planned maintenance
            elif event_type == 'Power failure':
                impact_minutes = np.random.uniform(20, 240)  # 20 minutes - 4 hours
                resolved_minutes = np.random.uniform(40, 300)  # 40 minutes - 5 hours
            else:
                impact_minutes = np.random.uniform(10, 90)  # 10-90 minutes
                resolved_minutes = np.random.uniform(20, 120)  # 20-120 minutes
            
            # Severity based on impact
            if impact_minutes > 180:
                severity = 'High'
            elif impact_minutes > 60:
                severity = 'Medium'
            else:
                severity = 'Low'
            
            event_data.append({
                'Event_ID': f"EV{str(i+1).zfill(6)}",
                'Train_ID': train_id,
                'Date': event_date.date(),
                'Time': event_date.time(),
                'Event_Type': event_type,
                'Severity': severity,
                'Impact_Minutes': round(impact_minutes, 1),
                'Resolved_Minutes': round(resolved_minutes, 1),
                'Affected_Routes': np.random.randint(1, 8),
                'Weather_At_Time': np.random.choice(self.weather_conditions),
                'Resolution_Method': np.random.choice(['Automatic', 'Manual', 'Emergency Protocol']),
                'Cost_Impact': round(np.random.uniform(1000, 100000), 2)  # Cost in currency
            })
        
        df = pd.DataFrame(event_data)
        df.to_csv('incident_event_data.csv', index=False)
        print(f"✓ Created incident_event_data.csv with {len(df)} records")
        return df
    
    def generate_rules_policies_data(self):
        """Generate Rules/Policies data for AI reference"""
        print("Generating Rules/Policies Data...")
        
        # Priority rules
        priority_rules = {
            'train_priority_hierarchy': [
                {'rank': 1, 'train_type': 'Rajdhani', 'priority_score': 100},
                {'rank': 2, 'train_type': 'Shatabdi', 'priority_score': 95},
                {'rank': 3, 'train_type': 'Express', 'priority_score': 80},
                {'rank': 4, 'train_type': 'Superfast', 'priority_score': 75},
                {'rank': 5, 'train_type': 'Passenger', 'priority_score': 60},
                {'rank': 6, 'train_type': 'Local', 'priority_score': 50},
                {'rank': 7, 'train_type': 'Metro', 'priority_score': 45},
                {'rank': 8, 'train_type': 'Freight', 'priority_score': 30}
            ],
            'emergency_override': True,
            'medical_emergency_priority': 100
        }
        
        # Delay tolerances
        delay_rules = {
            'max_allowed_delays': {
                'Rajdhani': {'max_delay_minutes': 15, 'compensation_threshold': 30},
                'Shatabdi': {'max_delay_minutes': 15, 'compensation_threshold': 30},
                'Express': {'max_delay_minutes': 30, 'compensation_threshold': 60},
                'Superfast': {'max_delay_minutes': 30, 'compensation_threshold': 60},
                'Passenger': {'max_delay_minutes': 45, 'compensation_threshold': 90},
                'Local': {'max_delay_minutes': 60, 'compensation_threshold': 120},
                'Metro': {'max_delay_minutes': 5, 'compensation_threshold': 15},
                'Freight': {'max_delay_minutes': 120, 'compensation_threshold': 240}
            }
        }
        
        # Safety buffers
        safety_rules = {
            'minimum_spacing_minutes': {
                'Same_Direction_Same_Track': 3,
                'Opposite_Direction_Single_Track': 5,
                'Junction_Crossing': 2,
                'High_Speed_Trains': 4,
                'Freight_Following_Passenger': 5
            },
            'speed_limits': {
                'Curved_Track': 80,
                'Bridge': 100,
                'Populated_Area': 60,
                'Station_Approach': 50,
                'Junction_Area': 40
            },
            'weather_restrictions': {
                'Fog': {'visibility_limit': '200m', 'max_speed': 60},
                'Storm': {'wind_limit': '60kmh', 'max_speed': 80},
                'Rain': {'max_speed': 100},
                'Snow': {'max_speed': 70}
            }
        }
        
        # Operational rules
        operational_rules = {
            'signal_protocols': {
                'Red': 'Complete Stop',
                'Yellow': 'Caution - Prepare to Stop',
                'Green': 'Proceed'
            },
            'maintenance_windows': {
                'Daily_Maintenance': '02:00-06:00',
                'Weekly_Major': 'Sunday 01:00-07:00',
                'Emergency_Closure': 'Immediate'
            },
            'crew_duty_limits': {
                'Continuous_Duty_Hours': 8,
                'Rest_Period_Hours': 10,
                'Monthly_Duty_Hours': 160
            }
        }
        
        # Combine all rules
        all_rules = {
            'priority_rules': priority_rules,
            'delay_rules': delay_rules,
            'safety_rules': safety_rules,
            'operational_rules': operational_rules,
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Save as JSON
        with open('rules_policies_data.json', 'w') as f:
            json.dump(all_rules, f, indent=2, default=str)
        
        print("✓ Created rules_policies_data.json")
        return all_rules
    
    def generate_complete_dataset(self):
        """Generate all tables in the complete dataset"""
        print("🚂 RAILWAY TRAFFIC CONTROL AI DATASET GENERATOR")
        print("=" * 60)
        print("Generating comprehensive synthetic dataset...")
        print()
        
        start_time = datetime.now()
        
        # Generate all tables
        train_df = self.generate_train_master_data(2000)
        junction_df, track_df = self.generate_track_junction_data(1000, 3000)
        signal_df = self.generate_signal_data(track_df, 5000)
        schedule_df = self.generate_schedule_data(train_df, junction_df, 50000)
        performance_df = self.generate_historical_performance_data(train_df, schedule_df, 500000)
        event_df = self.generate_incident_event_data(train_df, 50000)
        rules_data = self.generate_rules_policies_data()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print()
        print("=" * 60)
        print("🎯 DATASET GENERATION COMPLETED!")
        print(f"⏱️  Total time: {duration}")
        print()
        print("📊 Generated Files:")
        print("   • train_master_data.csv (2,000 trains)")
        print("   • junction_data.csv (1,000 junctions)")
        print("   • track_junction_data.csv (3,000 tracks)")
        print("   • signal_data.csv (5,000 signals)")
        print("   • schedule_timetable_data.csv (50,000 schedules)")
        print("   • historical_performance_data.csv (500,000 records)")
        print("   • incident_event_data.csv (50,000 events)")
        print("   • rules_policies_data.json (AI reference rules)")
        print()
        print("🤖 Ready for AI Training!")
        print("   Total records: ~610,000+ across all tables")
        print("   Realistic data patterns with proper relationships")
        print("   Indian railway context with actual station names")
        
        # Generate summary statistics
        self.generate_dataset_summary()
    
    def generate_dataset_summary(self):
        """Generate summary statistics of the dataset"""
        try:
            summary = {
                'generation_date': datetime.now().isoformat(),
                'total_files': 8,
                'total_records': 610000,
                'train_types_distribution': {},
                'weather_conditions': self.weather_conditions,
                'key_statistics': {
                    'stations_count': len(self.stations),
                    'train_types_count': len(self.train_types),
                    'max_train_speed': '180 km/h',
                    'date_range': '2021-2024 (3 years)',
                    'geographical_coverage': 'Pan-India'
                }
            }
            
            with open('dataset_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print("✓ Created dataset_summary.json")
            
        except Exception as e:
            print(f"Warning: Could not generate summary - {e}")

# Main execution
if __name__ == "__main__":
    generator = RailwayDatasetGenerator()
    generator.generate_complete_dataset()
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("1. Run this script: python railway_dataset_generator.py")
    print("2. Load CSV files into your AI training pipeline")
    print("3. Use rules_policies_data.json for AI decision logic")
    print("4. Tables are interconnected via ID fields (Train_ID, etc.)")
    print("="*60)