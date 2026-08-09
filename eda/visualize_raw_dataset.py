import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def load_dataset(config_path="config.json"):
    """Loads all CSVs defined in the config and merges them into one DataFrame."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    dfs = []
    for entry in config.get("csv_files", []):
        file_path = entry.get("file")
        dialect = entry.get("dialect", "Unknown")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # If CSV already has a Dialect column, keep it, otherwise use config
            if 'Dialect' not in df.columns:
                df['Dialect'] = dialect
            dfs.append(df)
        else:
            print(f"[Warning] Could not find {file_path}")
            
    if not dfs:
        raise ValueError("No data loaded. Check config.json and CSV paths.")
        
    data = pd.concat(dfs, ignore_index=True)
    # Clean up whitespace
    data['City'] = data['City'].str.strip()
    data['Category'] = data['Category'].str.strip()
    return data

def plot_dialect_category_split(df, output_path="plot_dialect_category.png"):
    """Plot 1: Dialect distribution split by Studio vs ITW (Stacked Bar)."""
    plt.figure(figsize=(10, 6))
    
    # Group by Dialect and Category
    counts = df.groupby(['Dialect', 'Category']).size().unstack(fill_value=0)
    
    # Plot stacked bar
    counts.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black', figsize=(10, 6))
    
    plt.title('Dataset Composition: Dialect by Environment', fontsize=14, fontweight='bold')
    plt.xlabel('Dialect', fontsize=12)
    plt.ylabel('Number of Source Videos', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Category')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_city_distribution(df, output_path="plot_city_distribution.png"):
    """Plot 2: City distribution per dialect."""
    plt.figure(figsize=(12, 6))
    
    # Count frequencies
    city_counts = df.groupby(['Dialect', 'City']).size().reset_index(name='Count')
    
    # Use seaborn to plot grouped bars
    sns.barplot(data=city_counts, x='Dialect', y='Count', hue='City', palette='tab20')
    
    plt.title('City Representation per Dialect', fontsize=14, fontweight='bold')
    plt.xlabel('Dialect', fontsize=12)
    plt.ylabel('Number of Source Videos', fontsize=12)
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def parse_duration(trim_str):
    """Calculates clip duration in seconds for the relevant bonus plot."""
    if pd.isna(trim_str) or not str(trim_str).strip():
        return None # Full videos are excluded from duration averages
    try:
        start_str, end_str = str(trim_str).split('-')
        def to_sec(ts):
            parts = ts.strip().split(':')
            sec = 0.0
            for part in parts:
                sec = sec * 60 + float(part)
            return sec
        return to_sec(end_str) - to_sec(start_str)
    except:
        return None

def plot_duration_variance(df, output_path="plot_duration_variance.png"):
    """Bonus Plot: Boxplot of clip durations per Dialect."""
    df_dur = df.copy()
    df_dur['Duration_Seconds'] = df_dur['Trim'].apply(parse_duration)
    df_dur = df_dur.dropna(subset=['Duration_Seconds'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_dur, x='Dialect', y='Duration_Seconds', hue='Category', palette='Set2')
    
    plt.title('Variance of Trimmed Clip Durations', fontsize=14, fontweight='bold')
    plt.xlabel('Dialect', fontsize=12)
    plt.ylabel('Duration (Seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def generate_interactive_map(df, output_path="dataset_map.html"):
    """Plot 3: Interactive HTML map of Romania/Moldova data coverage."""
    # Count total clips per city
    city_totals = df.groupby('City').size().reset_index(name='Total Videos')
    
    print("Geocoding cities (this requires internet access)...")
    geolocator = Nominatim(user_agent="thesis_dataset_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    lats, lons = [], []
    for city in city_totals['City']:
        # Search restricted to Romania and Moldova for accuracy
        query = f"{city}, Romania"
        if city.lower() in ["chisinau", "balti", "tiraspol", "orhei"]:
            query = f"{city}, Moldova"
            
        location = geocode(query)
        if location:
            lats.append(location.latitude)
            lons.append(location.longitude)
        else:
            print(f"[Warning] Could not geocode city: {city}")
            lats.append(None)
            lons.append(None)
            
    city_totals['lat'] = lats
    city_totals['lon'] = lons
    city_totals = city_totals.dropna()

    # Create an interactive scatter map
    fig = px.scatter_map(
        city_totals,
        lat="lat",
        lon="lon",
        size="Total Videos",
        color="Total Videos",
        hover_name="City",
        hover_data={"lat": False, "lon": False, "Total Videos": True},
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=30,
        zoom=5.5,
        center={"lat": 45.9432, "lon": 24.9668}, # Center of Romania
        title="Geographic Distribution of Source Media"
    )
    
    fig.update_layout(mapbox_style="carto-positron")
    fig.write_html(output_path)
    print(f"Saved interactive map: {output_path}")

if __name__ == "__main__":
    print("Loading dataset from config...")
    df = load_dataset()
    
    plot_dialect_category_split(df)
    plot_city_distribution(df)
    plot_duration_variance(df)
    generate_interactive_map(df)
    
    print("All visualizations complete. Download the files to view them.")