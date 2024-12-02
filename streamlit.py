import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import pytz
import networkx as nx
import plotly.graph_objects as g
import random
import base64
from streamlit_agraph import agraph, Node, Edge, Config
import colorsys
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

# Set the page configuration
st.set_page_config(layout="wide")  # Make the dashboard full-width

# Custom CSS to improve the sidebar and overall layout
st.markdown("""
    <style>
    /* Reduce padding and margins inside the sidebar */
    .css-1d391kg {  /* Adjust this class if needed */
        padding: 1rem 0.5rem;
    }
    /* Adjust margins and paddings to reduce spacing */
    .sidebar .stSelectbox, .sidebar .stDateInput {
        margin-bottom: 0.5rem;
    }
    /* Adjust the font size of labels and options in the sidebar */
    .sidebar .stSelectbox label, .sidebar .stDateInput label {
        font-size: 14px;
    }
    /* Adjust the font size of options inside the widgets */
    .sidebar .stSelectbox div[data-baseweb="select"] * {
        font-size: 12px;
    }
    /* Adjust the expander header */
    .streamlit-expanderHeader {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('middle_east_14_24.csv')
    data['event_date'] = pd.to_datetime(data['event_date'])
    return data

data = load_data()

# Handle NaN values
data.dropna(subset=['country', 'disorder_type', 'sub_event_type', 'event_date'], inplace=True)
data['actor1'] = data['actor1'].fillna('Unknown')
data['actor2'] = data['actor2'].fillna('Unknown')
data['notes'] = data['notes'].fillna('No description available.')

@st.cache_data
def fetch_newsdata_top_headlines(selected_countries):
    """
    Fetch top political headlines for selected countries using the NewsData API.
    Returns a dictionary where keys are country codes and values are DataFrames of news articles.
    """
    api_key = "pub_60374bbf67928ff470e6b42e67f06730de270"  # Replace with your API key
    base_url = "https://newsdata.io/api/1/latest"
    headlines = {}

    for country_code in selected_countries:
        params = {
            "apikey": api_key,
            "category": "politics",
            "country": country_code,
            "language": "fr,en" 
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get("results", [])
            headlines[country_code] = pd.DataFrame(articles)
        else:
            st.error(f"Error fetching news for {country_code}: {response.status_code}")
            headlines[country_code] = pd.DataFrame()  # Empty DataFrame for errors

    return headlines

def fetch_weather_forecast(location):
    """
    Fetch weather forecast for a given location using Tomorrow.io API.
    """
    api_key = "FxBvuJRSXWixNasoRGImQNmaZbevW69d"  # Replace with your actual API key
    base_url = f"https://api.tomorrow.io/v4/weather/forecast?location={location}&apikey={api_key}"
    headers = {"accept": "application/json"}
    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        weather_data = response.json()
        return weather_data.get("timelines", {}).get("hourly", [])
    else:
        st.error(f"Error fetching weather data: {response.status_code}")
        return []

# Define predefined cities and their corresponding countries
city_mapping = {
    "Tel Aviv": ("Israel", "Asia/Jerusalem"),
    "Haifa": ("Israel", "Asia/Jerusalem"),
    "Damascus": ("Syria", "Asia/Damascus"),
    "Idlib": ("Syria", "Asia/Damascus"),
    "Saida": ("Lebanon", "Asia/Beirut"),
    "Beirut": ("Lebanon", "Asia/Beirut"),
    "West Bank": ("Palestine", "Asia/Hebron"),
    "Gaza": ("Palestine", "Asia/Hebron")
}

def is_daytime_now(local_time):
    """
    Determine if it's currently daytime based on local time.

    Args:
        local_time (datetime): Current local time.

    Returns:
        bool: True if daytime, False if nighttime.
    """
    sunrise_hour = 6   # 6 AM
    sunset_hour = 18   # 6 PM

    if sunrise_hour <= local_time.hour < sunset_hour:
        return True
    else:
        return False

def get_weather_emoji(weather_code, is_daytime=True):
    """
    Map weather codes to emojis based on daytime or nighttime.

    Args:
        weather_code (int): The weather code from the API.
        is_daytime (bool): Flag indicating if it's daytime.

    Returns:
        str: Corresponding weather emoji.
    """
    # Define emoji mappings for daytime conditions
    day_emojis = {
        1000: "☀️ Sunny",
        1001: "🌤️ Partly Cloudy",
        1100: "🌥️ Mostly Cloudy",
        1101: "☁️ Cloudy",
        2000: "🌫️ Fog",
        2100: "🌫️ Light Fog",
        3000: "🌧️ Light Rain",
        3001: "🌧️ Rain",
        5000: "🌨️ Light Snow",
        5001: "🌨️ Snow",
        6000: "🌨️ Freezing Rain",
        6001: "🌨️ Snow Showers",
        7000: "🌪️ Windy",
        7101: "🌪️ Strong Winds",
        8000: "🌩️ Thunderstorm",
        # Add more mappings as needed
    }

    # Define emoji mappings for nighttime conditions
    night_emojis = {
        1000: "🌙 Clear Night",
        1001: "🌤️ Partly Cloudy Night",
        1100: "🌥️ Mostly Cloudy Night",
        1101: "☁️ Cloudy Night",
        2000: "🌫️ Fog",
        2100: "🌫️ Light Fog",
        3000: "🌧️ Light Rain",
        3001: "🌧️ Rain",
        5000: "🌨️ Light Snow",
        5001: "🌨️ Snow",
        6000: "🌨️ Freezing Rain",
        6001: "🌨️ Snow Showers",
        7000: "🌪️ Windy",
        7101: "🌪️ Strong Winds",
        8000: "🌩️ Thunderstorm",
        # Add more mappings as needed
    }

    if is_daytime:
        return day_emojis.get(weather_code, "🌫️ Unknown")
    else:
        return night_emojis.get(weather_code, "🌫️ Unknown")

# Title with better styling
st.title('🌍 Middle East Events Dashboard')

# Sidebar with filters as dropdown lists
with st.sidebar:
    st.header('Filters')

    # Location Filter with individual checkboxes
    st.subheader("Select Countries")
    countries = sorted(data['country'].dropna().unique())
    selected_countries = []

    for country in countries:
        if st.checkbox(country, value=True):  # Default to all selected
            selected_countries.append(country)

    if selected_countries:
        filtered_data = data[data['country'].isin(selected_countries)]
    else:
        filtered_data = data.copy()  # If no country is selected, show all data

    # Disorder Type Filter
    disorder_types = sorted(filtered_data['disorder_type'].dropna().unique())
    selected_disorder_type = st.selectbox('Select Disorder Type', ['All'] + disorder_types)
    if selected_disorder_type != 'All':
        filtered_data = filtered_data[filtered_data['disorder_type'] == selected_disorder_type]

    # Sub-Event Type Filter
    sub_event_types = sorted(filtered_data['sub_event_type'].dropna().unique())
    selected_sub_event_type = st.selectbox('Select Sub-Event Type', ['All'] + sub_event_types)
    if selected_sub_event_type != 'All':
        filtered_data = filtered_data[filtered_data['sub_event_type'] == selected_sub_event_type]

    # Date Range Filter
    min_date = data['event_date'].min()
    max_date = data['event_date'].max()
    start_date, end_date = st.date_input(
        'Select Date Range',
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    filtered_data = filtered_data[
        (filtered_data['event_date'] >= pd.to_datetime(start_date)) &
        (filtered_data['event_date'] <= pd.to_datetime(end_date))
    ]

# Create tabs
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    '🏠 Home', '📰 Breaking News', '📍 Map', '📈 Timeline', '📋 Data', '⚠️ Fatalities',
    '🤝 Actor Analysis', '📝 Word Cloud of Event Descriptions'
])

country_mapping = {
    "Palestine":"ps",
    "Syria": "sy",
    "Lebanon": "lb",
    "Israel": "il"
    # Add more countries as needed
}

# Convert selected country names to codes
selected_country_codes = [country_mapping[country] for country in selected_countries if country in country_mapping]

# YouTube Videos List
youtube_videos = [
    {
        "title": "Israel-Hamas War Explained | Israel Vs Palestine Conflict Explained | Israel Gaza War Explained",
        "url": "https://www.youtube.com/watch?v=Xxb1a8k0O2I"
    },
    {
        "title": "How Lebanon Descended Into Civil War | Lebanon Documentary",
        "url": "https://www.youtube.com/watch?v=q70bGTwk4VM"
    },
    {
        "title": "History of Israel Documentary",
        "url": "https://www.youtube.com/watch?v=lyhW8AkkZdw"
    },
    {
        "title": "Why Syria’s Civil War Has Restarted",
        "url": "https://www.youtube.com/watch?v=s4sTCKZajp8"
    },
    {
        "title": "What is happening with the Syrian Civil War? Syrian Civil War Explained | Syrian Conflict Explained",
        "url": "https://www.youtube.com/watch?v=q4ZVV-6FrAs"
    },
    {
        "title": "Israel and Lebanon’s history of conflict explained",
        "url": "https://www.youtube.com/watch?v=Hjr6qVRNDbE"
    },

    {
        "title": "How Palestinians were expelled from their homes",
        "url": "https://www.youtube.com/watch?v=rGVgjS98OsU"
    },    
    {
        "title": "Who owns Jerusalem? | DW Documentary",
        "url": "https://www.youtube.com/watch?v=kUMYT6tozEg"
    },    
    {
        "title": "History of the ancient Levant",
        "url": "https://www.youtube.com/watch?v=h-Fgpd63MzM"
    },    
    {
        "title": "Origins of the Word Palestine",
        "url": "https://www.youtube.com/watch?v=BHsqVB9nxFY"
    },                
    # Add more videos as needed
]

# Weather tab (tab0)
with tab0:
    st.markdown("<h1 style='text-align: center; font-size: 36px;'>🌦️ Weather Forecast</h1>", unsafe_allow_html=True)
    st.write("Select a city to get the current weather forecast.")

    # Dropdown for city selection
    selected_city = st.selectbox("Select a city:", options=list(city_mapping.keys()), index=0)

    # Fetch weather forecast for the selected city
    if selected_city:
        city_country, timezone = city_mapping[selected_city]
        weather_forecast = fetch_weather_forecast(selected_city)

        if weather_forecast:
            # Extract the first available weather data
            first_weather = weather_forecast[0]
            temperature = round(first_weather["values"]["temperature"], 1)
            weather_code = first_weather["values"]["weatherCode"]

            # Get the current local time for the selected city's timezone
            local_time = datetime.now(pytz.timezone(timezone))
            date = local_time.strftime("%A, %d %B %Y")
            time = local_time.strftime("%I:%M %p")

            # Determine if it's daytime
            daytime = is_daytime_now(local_time)

            # Get the appropriate weather emoji
            weather_emoji = get_weather_emoji(weather_code, is_daytime=daytime)

            # Aesthetic Display
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; margin-top: 30px; font-family: Arial, sans-serif;">
                    <div style="text-align: center; margin-right: 40px;">
                        <h1 style="font-size: 80px; margin: 0;">🕒</h1>
                        <h2 style="font-size: 28px; margin: 5px; color: #555;">{date}</h2>
                        <h2 style="font-size: 36px; margin: 5px; color: #333;">{time}</h2>
                    </div>
                    <div style="text-align: center; margin-left: 40px;">
                        <h1 style="font-size: 36px; margin: 0; color: #555;">{selected_city}</h1>
                        <h2 style="font-size: 80px; margin: 5px; color: #333;">{weather_emoji}</h2>
                        <h2 style="font-size: 40px; margin: 5px; color: #333;">{temperature}°C</h2>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.write(f"No weather data available for {selected_city}. Please try again later.")

    # Add YouTube Videos Below Weather Information
    st.markdown("## 🎥 Recommended Videos")
    
    # Method 1: Using st.video (Recommended for simplicity and responsiveness)
    for video in youtube_videos:
        st.subheader(video["title"])
        # Create three columns: left spacer, video, right spacer
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(video["url"], start_time=0)

    # OR

    # Method 2: Using HTML iframe for customization
    # for video in youtube_videos:
    #     st.subheader(video["title"])
    #     st.markdown(
    #         f"""
    #         <iframe width="700" height="400" src="https://www.youtube.com/embed/{video['url'].split('v=')[-1]}"
    #         title="{video['title']}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    #         allowfullscreen></iframe>
    #         """,
    #         unsafe_allow_html=True
    #     )

with tab1:
    st.header('📰 Breaking News')
    st.write('Latest political news articles based on the selected countries.')

    if selected_country_codes:
        # Fetch top political headlines for the selected country codes
        news_by_country = fetch_newsdata_top_headlines(selected_country_codes)

        for country_code, df_news in news_by_country.items():
            if not df_news.empty:
                st.subheader(f"Top Political Headlines for {country_code.upper()}")
                for index, row in df_news.head(3).iterrows():  # Limit to 3 articles per country
                    st.subheader(row['title'])
                    st.write(f"**Source:** {row.get('source_id', 'Unknown')}  |  **Published:** {row.get('pubDate', 'Unknown')}")
                    st.write(row.get('description', 'No description available.'))
                    st.markdown(f"[Read more]({row.get('link', '#')})")
                    st.markdown("---")
            else:
                st.write(f"No news articles available for {country_code.upper()}.")
    else:
        st.write("Please select one or more countries in the sidebar to view breaking news.")

with tab2:
    st.header('Interactive Map')

    st.subheader('Fatalities by Location')
    # Aggregate fatalities by location
    location_fatalities = filtered_data.groupby(['latitude', 'longitude', 'location']).agg({'fatalities': 'sum'}).reset_index()
    location_fatalities = location_fatalities[location_fatalities['fatalities'] > 0]

    # Dynamically update the map title based on selected countries
    if selected_countries:
        countries_title = ', '.join(selected_countries)
        map_title = f"Fatalities in {countries_title}"
    else:
        map_title = "Fatalities in All Countries"

    if not location_fatalities.empty:
        fig_bubble_map = px.scatter_mapbox(
            location_fatalities,
            lat='latitude',
            lon='longitude',
            size='fatalities',
            hover_name='location',
            hover_data={'fatalities': True},
            color='fatalities',
            color_continuous_scale=px.colors.sequential.Cividis,
            size_max=15,
            zoom=5,
            mapbox_style='carto-positron',
            title=map_title  # Use the dynamic title here
        )
        fig_bubble_map.update_layout(margin={'l': 0, 'r': 0, 't': 50, 'b': 0})
        st.plotly_chart(fig_bubble_map, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")

with tab3:
    st.header('Event Timeline')

    # Create a more detailed timeline with better height and aggregation
    col1, col2 = st.columns([1, 4])

    with col1:
        timeline_interval = st.selectbox(
            'Aggregate by',
            ['Day', 'Week', 'Month'],
            index=1
        )

    # Aggregate data based on selected interval
    if timeline_interval == 'Day':
        timeline_data = filtered_data.groupby(['event_date', 'country']).size().reset_index(name='counts')
    elif timeline_interval == 'Week':
        filtered_data['week'] = filtered_data['event_date'].dt.isocalendar().week
        timeline_data = filtered_data.groupby([pd.Grouper(key='event_date', freq='W'), 'country']).size().reset_index(name='counts')
    else:
        timeline_data = filtered_data.groupby([pd.Grouper(key='event_date', freq='M'), 'country']).size().reset_index(name='counts')

    if not timeline_data.empty:
        fig_timeline = px.area(
            timeline_data,
            x='event_date',
            y='counts',
            color='country',
            title=f'Event Timeline (Aggregated by {timeline_interval})',
            labels={'event_date': 'Date', 'counts': 'Number of Events'},
            height=500  # Increased height
        )

        fig_timeline.update_layout(
            xaxis=dict(title='Date', gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(title='Number of Events', gridcolor='rgba(0,0,0,0.1)'),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")

with tab4:
    st.header('Event Details')
    # Show only relevant columns
    display_columns = ['event_date', 'country', 'location', 'sub_event_type', 'actor1', 'actor2', 'fatalities']
    st.dataframe(filtered_data[display_columns], height=400)

with tab5:
    st.header('⚠️ Fatalities Analysis')

    if not filtered_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Total fatalities by country
            fatalities_by_country = (
                filtered_data.groupby('country')['fatalities']
                .sum()
                .reset_index()
                .astype({'fatalities': int})  # Convert fatalities to integer
            )
            fig_fatalities_country = px.bar(
                fatalities_by_country,
                x='country',
                y='fatalities',
                title='Total Fatalities by Country',
                color='country',
                labels={'fatalities': 'Total Fatalities', 'country': 'Country'}
            )
            fig_fatalities_country.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_fatalities_country, use_container_width=True)

        with col2:
            # Fatalities over time
            fatalities_over_time = (
                filtered_data.groupby(pd.Grouper(key='event_date', freq='W'))['fatalities']
                .sum()
                .reset_index()
                .astype({'fatalities': int})  # Convert fatalities to integer
            )
            fig_fatalities_time = px.line(
                fatalities_over_time,
                x='event_date',
                y='fatalities',
                title='Weekly Fatalities Over Time',
                labels={'event_date': 'Date', 'fatalities': 'Fatalities'}
            )
            fig_fatalities_time.update_layout(height=400)
            st.plotly_chart(fig_fatalities_time, use_container_width=True)
        
        # Top 10 events with highest fatalities
        st.subheader('🏆 Top 10 Events with Highest Fatalities')
        top_fatalities = (
            filtered_data.nlargest(10, 'fatalities')[['event_date', 'country', 'sub_event_type', 'fatalities', 'notes']]
            .astype({'fatalities': int})  # Convert fatalities to integer
        )

        if not top_fatalities.empty:
            for index, row in top_fatalities.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"### {row['fatalities']} Fatalities")
                        st.markdown(f"**Date:** {row['event_date'].strftime('%Y-%m-%d')}")
                    with col2:
                        st.markdown(f"**Country:** {row['country']}")
                        st.markdown(f"**Event Type:** {row['sub_event_type']}")
                        st.markdown(f"**Details:** {row['notes']}")
                        st.markdown("---")  # Divider between events
        else:
            st.write("No data available for the selected filters.")
    else:
        st.write("No data available for the selected filters.")
with tab6:
    st.header('🤝 Interactive Actor Analysis')

    st.subheader('Actor Interactions Directed Graph')

    if not filtered_data.empty:
        # Simplify actor names
        def simplify_actor(actor):
            if pd.isna(actor) or actor.strip() == '':
                return 'Unknown'
            else:
                return actor.strip()

        filtered_data['actor1_simplified'] = filtered_data['actor1'].apply(simplify_actor)
        filtered_data['actor2_simplified'] = filtered_data['actor2'].apply(simplify_actor)

        # Get top actors to limit the number of nodes
        top_n = 10  # Adjust the number of top actors to include
        top_actor1 = filtered_data['actor1_simplified'].value_counts().nlargest(top_n).index.tolist()
        top_actor2 = filtered_data['actor2_simplified'].value_counts().nlargest(top_n).index.tolist()

        # Combine top actors from both actor1 and actor2
        top_actors = list(set(top_actor1 + top_actor2))

        # Filter data to include only top actors
        filtered_top_data = filtered_data[
            (filtered_data['actor1_simplified'].isin(top_actors)) &
            (filtered_data['actor2_simplified'].isin(top_actors))
        ]

        # Re-aggregate filtered data for the graph
        graph_df = filtered_top_data.groupby(['actor1_simplified', 'actor2_simplified']).size().reset_index(name='count')

        # Create a NetworkX directed graph
        G = nx.DiGraph()

        for _, row in graph_df.iterrows():
            source = row['actor1_simplified']
            target = row['actor2_simplified']
            count = row['count']
            G.add_edge(source, target, weight=count)

        # Generate a list of unique pastel colors
        def generate_unique_pastel_colors(n):
            colors = []
            for i in range(n):
                hue = random.uniform(0, 1)
                saturation = random.uniform(0.3, 0.6)  # Lower saturation for pastel
                lightness = random.uniform(0.7, 0.9)   # Higher lightness for pastel
                rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                rgb = tuple(int(x * 255) for x in rgb)
                colors.append('#%02x%02x%02x' % rgb)
            return colors

        unique_nodes = list(G.nodes())
        pastel_colors = generate_unique_pastel_colors(len(unique_nodes))
        node_color_map = dict(zip(unique_nodes, pastel_colors))

        # Initialize PyVis network
        net = Network(height="600px", width="100%", directed=True, notebook=False, bgcolor="#ffffff", font_color="black")

        # Customize physics for better spacing
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.01,
            damping=0.09
        )

        # Add nodes with customization
        for node in unique_nodes:
            net.add_node(
                node, 
                label=node, 
                title=node,
                color=node_color_map[node],
                size=25,  # Adjusted node size for better aesthetics
                font={'size': 14, 'color': 'black'},
                physics=True  # Enable physics for individual node movement
            )

        # Add edges with customization
        for source, target, data in G.edges(data=True):
            net.add_edge(
                source, 
                target, 
                value=data['weight'],
                title=f"Interactions: {data['weight']}",
                color='#AAAAAA',  # Neutral grey for edges
                smooth=True  # Smooth edges for better aesthetics
            )

        # Customize node appearance further with shadows
        for node in net.nodes:
            node['shadow'] = {
                "enabled": True,
                "color": "#000000",
                "size": 10,
                "x": 3,
                "y": 3
            }

        # Set additional options for better interactivity and aesthetics
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 14,
              "color": "black"
            },
            "scaling": {
              "min": 10,
              "max": 30
            },
            "shape": "dot"
          },
          "edges": {
            "color": {
              "color": "#AAAAAA",
              "highlight": "#FF0000"
            },
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal",
              "roundness": 0.4
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 250,
              "springConstant": 0.01,
              "damping": 0.09
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          }
        }
        """)

        # Generate the HTML content
        try:
            # For newer versions of pyvis
            html_content = net.generate_html()
        except AttributeError:
            # If generate_html is not available, use to_html
            html_content = net.to_html()

        # Embed the HTML in Streamlit
        components.html(html_content, height=600, width=800)

        # Top Actors Bar Chart
        st.subheader('Top Actors')
        top_actors_counts = filtered_data['actor1_simplified'].value_counts().reset_index()
        top_actors_counts.columns = ['Actor', 'Event Count']
        fig_top_actors = px.bar(
            top_actors_counts.head(10),  # Show top 10 actors
            x='Actor',
            y='Event Count',
            title='Top Actors by Number of Events',
            labels={'Event Count': 'Number of Events', 'Actor': 'Actor'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_top_actors.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top_actors, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")

with tab7:
    st.header('📝 Word Cloud of Event Descriptions')

    if not filtered_data.empty:
        # Combine all notes into one string
        text = ' '.join(filtered_data['notes'].dropna().tolist())

        # Generate a word cloud object
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=200
        ).generate(text)

        # Display the generated image
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No data available for the selected filters.")

# Footer
st.sidebar.markdown('---')
st.sidebar.markdown(f'*Data last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')
