import streamlit as st
import pandas as pd
import numpy as np
import datetime
import folium
from streamlit_folium import folium_static
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner appearance
st.markdown("""
<style>
    .main {
        max-width: 800px;
        padding: 1rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        width: 100%;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
        text-align: center;
        color: #000000;  /* Ensures black text */
    }
    .prediction-box h2 {
        color: #000000 !important;  /* Forces black color for the heading */
    }
    .info-text {
        color: #000000;  /* Black color for the tip text */
        font-size: 0.9em;
    }
    /* Ensure all other text is black */
    body {
        color: #000000 !important;
    }
    .stTextInput, .stNumberInput, .stSelectbox, .stTimeInput {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# NYC borough coordinates
borough_coords = {
    'Manhattan': (40.7831, -73.9712),
    'Brooklyn': (40.6782, -73.9442),
    'Queens': (40.7282, -73.7949),
    'Bronx': (40.8448, -73.8648),
    'Staten Island': (40.5795, -74.1502),
    'JFK Airport': (40.6413, -73.7781),
    'LaGuardia Airport': (40.7769, -73.8740)
}

# Fare calculation function
def calculate_fare(distance, duration, pickup_loc, dropoff_loc, passengers, hour, tip=0):
    """Calculate taxi fare based on NYC TLC rules with realistic pricing"""
    # Base fare
    base_fare = 3.00
    
    # Distance rate (per mile)
    distance_rate = 2.50
    
    # Time rate (per minute when speed <12mph)
    time_rate = 0.50
    
    # Calculate estimated time if not provided (assuming average speed of 12mph in NYC)
    if duration == 0:
        duration = (distance / 12) * 60  # in minutes
    
    # Calculate metered fare
    metered_fare = base_fare + (distance * distance_rate)
    
    # Add time charge (only applies when speed <12mph)
    if (distance / (duration/60)) < 12:  # if speed <12mph
        metered_fare += (duration * time_rate)
    
    # Add NY State Tax ($0.50) and MTA Tax ($0.50)
    metered_fare += 1.00
    
    # Add peak hour surcharge (4-8pm weekday, $2.50)
    if hour >= 16 and hour <= 20 and datetime.datetime.now().weekday() < 5:
        metered_fare += 2.50
    
    # Add overnight surcharge (8pm-6am, $0.50)
    if hour >= 20 or hour < 6:
        metered_fare += 0.50
    
    # Add airport surcharge
    if pickup_loc in ['JFK Airport', 'LaGuardia Airport']:
        metered_fare += 2.50
    if dropoff_loc in ['JFK Airport', 'LaGuardia Airport']:
        metered_fare += 2.50
    
    # Add tolls for certain routes
    if (pickup_loc == 'Manhattan' and dropoff_loc == 'Brooklyn') or \
       (pickup_loc == 'Brooklyn' and dropoff_loc == 'Manhattan'):
        metered_fare += 4.00  # Approx bridge toll
    
    # Total fare
    total_fare = metered_fare + tip
    
    return {
        'base_fare': 3.00,
        'distance_charge': distance * distance_rate,
        'time_charge': (duration * time_rate) if (distance / (duration/60)) < 12 else 0,
        'taxes_surcharges': 1.00 + (2.50 if (hour >= 16 and hour <= 20 and datetime.datetime.now().weekday() < 5) else 0) + (0.50 if (hour >= 20 or hour < 6) else 0),
        'airport_charges': (2.50 if pickup_loc in ['JFK Airport', 'LaGuardia Airport'] else 0) + (2.50 if dropoff_loc in ['JFK Airport', 'LaGuardia Airport'] else 0),
        'tolls': 4.00 if (pickup_loc == 'Manhattan' and dropoff_loc == 'Brooklyn') or (pickup_loc == 'Brooklyn' and dropoff_loc == 'Manhattan') else 0,
        'tip': tip,
        'total': total_fare
    }

# Main app
st.title("NYC Taxi Fare Calculator")
st.markdown("Estimate your taxi fare in New York City based on trip details.")

# Input form
with st.form("fare_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pickup_loc = st.selectbox("Pickup Location", list(borough_coords.keys()))
        distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=50.0, value=2.5, step=0.1)
        passengers = st.number_input("Passengers", min_value=1, max_value=6, value=1)
    
    with col2:
        dropoff_loc = st.selectbox("Dropoff Location", list(borough_coords.keys()))
        duration = st.number_input("Estimated Trip Time (minutes)", min_value=0, max_value=180, value=0)
        tip = st.number_input("Tip Amount ($)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
    
    pickup_time = st.time_input("Pickup Time", datetime.time(12, 0))
    
    submitted = st.form_submit_button("Calculate Fare")

# Calculate and display results
if submitted:
    hour = pickup_time.hour
    fare_details = calculate_fare(distance, duration, pickup_loc, dropoff_loc, passengers, hour, tip)
    
    # Show map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    folium.Marker(borough_coords[pickup_loc], popup=f"Pickup: {pickup_loc}", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(borough_coords[dropoff_loc], popup=f"Dropoff: {dropoff_loc}", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([borough_coords[pickup_loc], borough_coords[dropoff_loc]], color="blue").add_to(m)
    
    st.subheader("Trip Route")
    folium_static(m, width=700, height=400)
    
    # Display fare breakdown
    st.subheader("Fare Breakdown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Fare", f"${fare_details['base_fare']:.2f}")
        st.metric("Distance Charge", f"${fare_details['distance_charge']:.2f}")
        st.metric("Time Charge", f"${fare_details['time_charge']:.2f}")
    
    with col2:
        st.metric("Taxes & Surcharges", f"${fare_details['taxes_surcharges']:.2f}")
        st.metric("Airport Charges", f"${fare_details['airport_charges']:.2f}")
        st.metric("Tolls", f"${fare_details['tolls']:.2f}")
    
    # Total fare
    st.markdown(f"""
    <div class="prediction-box">
        <h2>Estimated Total Fare: ${fare_details['total']:.2f}</h2>
        <p class="info-text">(Including ${fare_details['tip']:.2f} tip)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fare breakdown chart
    breakdown_data = {
        "Component": ["Base Fare", "Distance", "Time", "Taxes/Surcharges", "Airport Fees", "Tolls", "Tip"],
        "Amount": [
            fare_details['base_fare'],
            fare_details['distance_charge'],
            fare_details['time_charge'],
            fare_details['taxes_surcharges'],
            fare_details['airport_charges'],
            fare_details['tolls'],
            fare_details['tip']
        ]
    }
    
    fig = px.bar(
        breakdown_data,
        x="Component",
        y="Amount",
        color="Component",
        title="Fare Components Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig)
    
    # Additional info
    st.info("""
    **Note:** This is an estimate based on NYC TLC fare rules. Actual fare may vary based on:
    - Traffic conditions
    - Exact route taken
    - Additional stops or waiting time
    - Other applicable surcharges
    """)

# About section
st.sidebar.title("About")
st.sidebar.info("""
This calculator estimates NYC taxi fares based on:
- NYC TLC fare rules
- Distance traveled
- Time of day
- Pickup/dropoff locations
- Additional surcharges

Fares are calculated using standard NYC taxi rates.
""")
