import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def load_dataset(config_path="./config.json"):
    """Loads all CSVs defined in the config and merges them into one DataFrame."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    dfs = []
    for entry in config.get("csv_files", []):
        file_path = entry.get("file")
        dialect = entry.get("dialect", "Unknown")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Dialect' not in df.columns:
                df['Dialect'] = dialect
            dfs.append(df)
        else:
            print(f"[Warning] Could not find {file_path}")
            
    if not dfs:
        raise ValueError("No data loaded. Check config.json and CSV paths.")
        
    data = pd.concat(dfs, ignore_index=True)
    data['City'] = data['City'].str.strip()
    data['Category'] = data['Category'].str.strip()
    return data

def plot_dialect_category_split(df, output_path="plot_dialect_category.png"):
    """Plot 1: Dialect distribution split by Studio vs ITW (Stacked Bar)."""
    plt.figure(figsize=(10, 6))
    counts = df.groupby(['Dialect', 'Category']).size().unstack(fill_value=0)
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

def plot_city_distribution_per_dialect(df):
    """Plot 2: Horizontal Plotly barcharts per dialect with hatched patterns."""
    dialects = df['Dialect'].unique()
    
    for dialect in dialects:
        subset = df[df['Dialect'] == dialect]
        counts = subset.groupby(['City', 'Category']).size().reset_index(name='Count')
        
        # Sort cities by total count so the longest bars are at the top
        order = counts.groupby('City')['Count'].sum().sort_values(ascending=True).index
        
        fig = px.bar(
            counts,
            x='Count',
            y='City',
            color='Category',
            pattern_shape='Category', 
            orientation='h',
            barmode='stack',
            title=f'City Distribution: {dialect} Dialect',
            category_orders={'City': order}
        )
        
        fig.update_layout(
            xaxis_title="Number of Videos",
            yaxis_title="City",
            font=dict(size=14),
            plot_bgcolor='white',
            margin=dict(l=100, r=20, t=50, b=50)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        html_out = f"plot_city_{dialect}.html"
        fig.write_html(html_out)
        print(f"Saved interactive bar chart: {html_out}")
        
        # Try to save as PNG for thesis inclusion (requires kaleido)
        try:
            png_out = f"plot_city_{dialect}.png"
            fig.write_image(png_out, scale=2)
            print(f"Saved static image: {png_out}")
        except ValueError:
            pass

def parse_duration(trim_str):
    """Calculates clip duration in seconds."""
    if pd.isna(trim_str) or not str(trim_str).strip():
        return None
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
    """Plot 3: Interactive HTML map of Romania/Moldova with color, size, and labels."""
    # Group by City AND Dialect to apply color coding
    city_totals = df.groupby(['City', 'Dialect']).size().reset_index(name='Total Videos')
    
    print("Geocoding cities (this requires internet access)...")
    geolocator = Nominatim(user_agent="thesis_dataset_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    lats, lons = [], []
    for city in city_totals['City']:
        query = f"{city}, Romania"
        if city.lower() in ["chisinau", "balti", "tiraspol", "orhei", "cahul", "ungheni"]:
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

    fig = px.scatter_map(
        city_totals,
        lat="lat",
        lon="lon",
        size="Total Videos",
        color="Dialect",
        text="City",
        hover_name="City",
        hover_data={"lat": False, "lon": False, "Total Videos": True, "Dialect": True},
        size_max=35,
        zoom=5.8,
        center={"lat": 46.5, "lon": 26.0},
        title="Geographic Distribution of Source Media"
    )
    
    # Updated for Dark Mode: White text so it shows up on the dark map
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=13, color='white', family='Arial') 
    )
    
    # Updated for Dark Mode: Switch the base map style and background colors
    fig.update_layout(
        map_style="carto-darkmatter",     # <-- Changed from mapbox_style to map_style
        margin={"r":0,"t":40,"l":0,"b":0},
        paper_bgcolor="#111111",          # Dark background around the map edges
        font=dict(color="white")          # Makes the title and legend text white
    )
    
    fig.write_html(output_path)
    print(f"Saved interactive map: {output_path}")

if __name__ == "__main__":
    print("Loading dataset from config...")
    df = load_dataset()
    
    plot_dialect_category_split(df)
    plot_city_distribution_per_dialect(df)
    plot_duration_variance(df)
    generate_interactive_map(df)
    
    print("All visualizations complete. Download the files to view them.")