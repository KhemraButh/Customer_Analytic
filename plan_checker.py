import nest_asyncio
import asyncio
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
import pandas as pd
import re
from datetime import datetime
import streamlit as st
import plotly.express as px
import pytz
from rapidfuzz import fuzz, process
import time
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
import numpy as np
#import sqlite3
import sqlite3
from folium.plugins import HeatMap
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# === MUST BE THE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="Sales Performance Dashboard", layout="wide", page_icon="📊"
)

nest_asyncio.apply()

# Your credentials
api_id = 20056320
api_hash = "4b1394e0f07625a3c25ea32fa3030218"
phone_number = os.environ["PHONE_NUMBER"]
target = ['https://t.me/+JeQdy_3JC20wYTY1']
session_name = "customer_session_2"

# === Custom CSS for beautiful styling ===
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .function-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border-left: 5px solid #2E8B57;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f8ff 0%, #e0f0e0 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .highlight {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .tab-content {
        padding: 20px;
        background: white;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .header-style {
        background: linear-gradient(90deg, #2E8B57 0%, #3CB371 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 15px 0;
        font-size: 1.3em;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

patterns = {
    "Name":              r"Name:\s*([^\n]*)",
    "Tel":               r"Tel:\s*([^\n]*)",
    "Business":          r"Business:\s*([^\n]*)",
    "Bank":              r"Bank:\s*([^\n]*)",
    "Amount":            r"Amount:\s*([^\n]*)",
    "Interest":          r"Interest:\s*([^\n]*)",
    "Loan Type":         r"Loan\s*Type:\s*([^\n]*)",
    "Tenure":            r"Tenure:\s*([^\n]*)",
    "Maturity":          r"Maturity:\s*([^\n]*)",
    "Potential H/M/L":   r"Potential\s*H/M/L:\s*([^\n]*)",
    "Potential Product": r"Potential\s*Product:\s*([^\n]*)"
}
import base64
import os

def get_base64_encoded_image(image_path):
    """Get base64 encoded string of an image"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Create three columns for the header
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])

with header_col1:
    try:
        #logo_path = "Logo-CMCB-15.png"
        logo_path = os.path.join(BASE_DIR, "Logo-CMCB_FA-15.png") 
        if os.path.exists(logo_path):
            logo_base64 = get_base64_encoded_image(logo_path)
            st.markdown(
                f"""
                <div style="background: white; padding: 10px; border-radius: 12px; 
                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); 
                            display: flex; align-items: center; justify-content: center;">
                    <img src="data:image/png;base64,{logo_base64}" 
                         width="100" style="border-radius: 8px;">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div style="background: #f0f0f0; padding: 20px; border-radius: 12px; 
                        text-align: center; color: #666;">
                <p style="margin: 0;">🏦<br>Logo</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown("""
        <div style="background: #f0f0f0; padding: 20px; border-radius: 12px; 
                    text-align: center; color: #666;">
            <p style="margin: 0;">🏦<br>Logo</p>
        </div>
        """, unsafe_allow_html=True)

with header_col2:
    st.markdown(
        """
        <div style="text-align: center; padding: 15px;">
            <h1 style="color: #004A08; margin: 0; font-size: 2.2rem; font-weight: 700;">
                Planing, Execution and Customer Data Management
            </h1>
            <p style="color: #2E8B57; margin: 5px 0 0 0; font-size: 1.1rem; font-weight: 500;">
                Performance & Execution Management System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_col3:
    st.markdown("")


# === Extract info from text ===
def extract_info_from_text(text):
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            # ✅ Extra guard: if value looks like a field name, treat as blank
            if re.match(
                r"^(Name|Tel|Business|Bank|Amount|Interest|Loan Type|Tenure|Maturity|Potential H/M/L|Potential Product)",
                value,
                re.IGNORECASE
            ):
                value = ""
            data[key] = value
        else:
            data[key] = ""
    return data

# === Scrape Telegram data ===
async def scrape_telegram_data(min_date, now):
    try:
        async with TelegramClient(session_name, api_id, api_hash) as client:
            await client.start(phone=phone_number)
            if not await client.is_user_authorized():
                st.error("Please check your phone for verification code")
                return None

            entity = await client.get_entity(target)
            
            # Fetch messages
            history = await client(
                GetHistoryRequest(
                    peer=entity,
                    limit=1000,
                    offset_date=None,
                    offset_id=0,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0,
                )
            )
            messages_data = []
            pending_customer = None
            cambodia_tz = pytz.timezone("Asia/Phnom_Penh")
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, msg in enumerate(reversed(history.messages)):
                # Update progress
                progress = (i + 1) / len(history.messages)
                progress_bar.progress(progress)
                status_text.text(
                    f"Processing message {i + 1} of {len(history.messages)}"
                )

                # Filter by date
                if hasattr(msg, "date") and msg.date:
                    msg_date = msg.date.astimezone(cambodia_tz)
                    if min_date.tzinfo is None:
                        min_date = cambodia_tz.localize(min_date)
                    if now.tzinfo is None:
                        now = cambodia_tz.localize(now)
                    if msg_date < min_date or msg_date > now:
                        continue
                else:
                    continue
                # Get sender info
                sender_name = None
                if getattr(msg, "from_id", None):
                    try:
                        sender = await client.get_entity(msg.from_id)
                        first = getattr(sender, "first_name", "") or ""
                        last = getattr(sender, "last_name", "") or ""
                        sender_name = (first + " " + last).strip() or getattr(
                            sender, "username", None
                        )
                    except Exception:
                        sender_name = None
                # Location messages
                if hasattr(msg, "geo") and msg.geo:
                    if pending_customer:
                        pending_customer.update(
                            {
                                "Latitude": msg.geo.lat,
                                "Longitude": msg.geo.long,
                                "Location_Date": msg_date,
                            }
                        )
                        messages_data.append(pending_customer)
                        pending_customer = None
                    continue
                # Text messages
                text = msg.message or getattr(msg, "caption", "") or ""
                if not text.strip():
                    continue
                extracted = extract_info_from_text(text)
                if any(extracted.values()):
                    customer_data = {
                        "Sender_Name": sender_name,
                        "Name": extracted.get("Name"),
                        "Tel": extracted.get("Tel"),
                        "Business": extracted.get("Business"),
                        "Bank": extracted.get("Bank"),
                        "Amount": extracted.get("Amount"),
                        "Interest": extracted.get("Interest"),
                        "Loan Type": extracted.get("Loan Type"),
                        "Tenure": extracted.get("Tenure"),
                        "Maturity": extracted.get("Maturity"),
                        "Potential_Level": extracted.get("Potential H/M/L"),
                        "Potential_Product": extracted.get("Potential Product"),
                        "Message_Date": msg_date,
                        "Latitude": None,
                        "Longitude": None,
                        "Location_Date": None
                    }
                    pending_customer = customer_data
            if pending_customer:
                messages_data.append(pending_customer)
            progress_bar.empty()
            status_text.empty()
            return messages_data
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

# Smart customer matching function
import re
#from fuzzywuzzy import fuzz, process

def smart_customer_matching(planned_customers, visited_customers, threshold=80):
    def preprocess_name(name):
        name = re.sub(r"[^\w\s]", "", str(name))
        name = re.sub(r"\s+", " ", name)
        return name.strip().lower()

    planned_processed = {preprocess_name(name): name for name in planned_customers}
    visited_processed = {preprocess_name(name): name for name in visited_customers}

    matched_pairs = {}
    unmatched_planned = set(planned_processed.keys())
    unmatched_visited = set(visited_processed.keys())

    for p_name in list(unmatched_planned):
        if not unmatched_visited:
            break  # no more visited customers to match

        best_match_result = process.extractOne(
            p_name, list(unmatched_visited), scorer=fuzz.token_sort_ratio
        )

        if best_match_result is None:
            continue

        # Depending on the library version, extractOne may return 2 or 3 items
        if len(best_match_result) >= 2:
            best_match = best_match_result[0]
            score = best_match_result[1]
        else:
            continue

        if score >= threshold:
            matched_pairs[planned_processed[p_name]] = visited_processed[best_match]
            unmatched_planned.remove(p_name)
            unmatched_visited.remove(best_match)

    unmatched_planned_original = {planned_processed[name] for name in unmatched_planned}
    unmatched_visited_original = {visited_processed[name] for name in unmatched_visited}

    return matched_pairs, unmatched_planned_original, unmatched_visited_original
# Helper function for map visualization
def create_customer_map(data):
    """Create an interactive map of customer locations"""
    if data.empty or 'Latitude' not in data.columns or 'Longitude' not in data.columns:
        st.warning("No location data available for mapping.")
        return None
    
    # Filter out rows with missing coordinates
    map_data = data.dropna(subset=['Latitude', 'Longitude'])
    
    if map_data.empty:
        st.warning("No valid coordinates found for mapping.")
        return None
    
    # Create base map centered on average coordinates
    avg_lat = map_data['Latitude'].mean()
    avg_lon = map_data['Longitude'].mean()
    
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    
    # Define color coding based on potential
    POTENTIAL_COLORS = {
        'H': 'red',      # High potential - Red
        'M': 'orange',   # Medium potential - Orange
        'L': 'green',    # Low potential - Green
        '': 'gray'       # Unknown - Gray
    }
    
    # Add markers for each customer
    for _, row in map_data.iterrows():
        # Determine marker color based on potential
        potential = str(row.get('Potential', '')).strip().upper()
        color = POTENTIAL_COLORS.get(potential, 'gray')
        
        # Create popup content
        popup_html = f"""
        <div style='width: 250px; font-size: 12px;'>
            <h4>{row.get('Customer Name', 'Unknown')}</h4>
            <b>Business:</b> {row.get('Biz Type', 'N/A')}<br>
            <b>Potential:</b> {potential}<br>
            <b>Income:</b> ${row.get('Monthly Income', 'N/A')}<br>
            <b>Product Interest:</b> {row.get('Product Interest', 'N/A')}<br>
            <b>Phone:</b> {row.get('Phone Number', 'N/A')}<br>
            <b>Visit Date:</b> {row.get('Message Date', 'N/A')}
        </div>
        """
        
        # Add marker to map
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row.get('Customer Name', 'Unknown'),
            icon=folium.Icon(color=color, icon='user', prefix='fa')
        ).add_to(m)
    
    return m

# === Streamlit App ===
def main():
    # Initialize session state
    if 'show_map' not in st.session_state:
        st.session_state.show_map = False
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 12
    
    # Three main functions
    tab1, tab2, tab3 = st.tabs(["📋 Planning Check", "📍 Market Visit Presentation", "🌍 Customer Analytics"])
    
    with tab1:
        st.markdown(
            """
            <div class="function-card">
                <h2>📋 Sales Planning & Performance Tracking</h2>
                <p>Upload your sales plan and compare with actual customer visits from Telegram</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        uploaded_file = st.file_uploader(
            "📤 Upload Sales Team Plan (Excel file)",
            type=["xlsx"],
            help="File should contain 'Sales Name' and 'Customer' columns",
        )
    
        if uploaded_file:
            try:
                sales_plan_df = pd.read_excel(uploaded_file)
    
                if "Sales Name" not in sales_plan_df.columns or "Customer" not in sales_plan_df.columns:
                    st.error("❌ The Excel file must contain 'Sales Name' and 'Customer' columns")
                else:
                    # Metrics
                    total_planned = len(sales_plan_df)
                    unique_sales = sales_plan_df["Sales Name"].nunique()
                    col1, col2 = st.columns(2)
                    col1.markdown(f"<div class='metric-card'><h3>{total_planned}</h3><p>Planned Customer Visits</p></div>", unsafe_allow_html=True)
                    col2.markdown(f"<div class='metric-card'><h3>{unique_sales}</h3><p>Sales Team Members</p></div>", unsafe_allow_html=True)
    
                    # Date range
                    col1, col2 = st.columns(2)
                    start_date = col1.date_input("Start Date", datetime(2025, 9, 1))
                    end_date = col2.date_input("End Date", datetime.now())
    
                    if st.button("🚀 Scrape & Analyze Performance", type="primary"):
                        with st.spinner("🔄 Scraping Telegram data..."):
                            min_date = datetime.combine(start_date, datetime.min.time())
                            max_date = datetime.combine(end_date, datetime.max.time())
                            telegram_data = asyncio.run(scrape_telegram_data(min_date, max_date))
    
                        if telegram_data:
                            telegram_df = pd.DataFrame(telegram_data)
                            st.session_state.telegram_df = telegram_df
    
                            # Performance analysis
                            planned_customers = set(sales_plan_df["Customer"].str.strip().str.lower().dropna())
                            visited_customers = set(telegram_df["Name"].str.strip().str.lower().dropna())
    
                            # Safe unpacking
                            try:
                                matched_pairs, unmatched_planned, unmatched_visited = smart_customer_matching(planned_customers, visited_customers, threshold=80)
                            except Exception as e:
                                st.error(f"Error in matching: {e}")
                                matched_pairs, unmatched_planned, unmatched_visited = {}, set(), set()
    
                            expanded_visited_customers = visited_customers.union(set(matched_pairs.keys()))
                            matched_customers = planned_customers.intersection(expanded_visited_customers)
                            missed_customers = planned_customers - expanded_visited_customers
    
                            # Metrics
                            visit_rate = (len(matched_customers)/len(planned_customers))*100 if planned_customers else 0
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Planned Customers", len(planned_customers))
                            col2.metric("Visited Customers", len(matched_customers))
                            col3.metric("Visit Rate", f"{visit_rate:.1f}%", delta=(f"{visit_rate-100:.1f}%" if visit_rate < 100 else None))
    
                            # Visualization
                            fig = px.pie(values=[len(matched_customers), len(missed_customers)],
                                         names=["Visited", "Not Visited"],
                                         title="Customer Visit Performance",
                                         color_discrete_map={"Visited": "#2E8B57", "Not Visited": "#FF6B6B"})
                            st.plotly_chart(fig, use_container_width=True)
    
                            # Detailed results
                            with st.expander("📊 View Detailed Analysis"):
                                comparison_data = [{"Customer": c, "Status": "Visited" if c in expanded_visited_customers else "Not Visited"} for c in planned_customers]
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    with tab2:
        telegram_df = pd.read_excel("customers_parsed.xlsx")
        #telegram_df = pd.read_exel("customer_parsed.xlsx")
                # Ensure numeric columns are properly formatted
                
        if "Interest" in telegram_df.columns:
                    telegram_df["Interest"] = pd.to_numeric(
                        telegram_df["Interest"], errors="coerce"
                    )
                
        if "Amount" in telegram_df.columns:
                    telegram_df["Amount"] = pd.to_numeric(
                        telegram_df["Amount"], errors="coerce"
                    )
                
        if "Tenure" in telegram_df.columns:
                    telegram_df["Tenure"] = pd.to_numeric(
                        telegram_df["Tenure"], errors="coerce"
                    )
        if "Maturity" in telegram_df.columns:
                    telegram_df["Maturity"] = pd.to_numeric(
                        telegram_df["Maturity"], errors="coerce"
                    )
                
                # Create a display copy with formatted values
        display_df = telegram_df.copy()
                
                # Format numeric columns for display
        if "Monthly Income" in display_df.columns:
                    display_df["Monthly Income"] = display_df["Monthly Income"].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) and x != 0 else ""
                    )
                
        if "Amount" in display_df.columns:
                    display_df["Amount"] = display_df["Amount"].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) and x != 0 else ""
                    )
                
        if "Interest" in display_df.columns:
                    display_df["Interest"] = display_df["Interest"].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) and x != 0 else ""
                    )
                
        if "Tenure" in display_df.columns:
                    display_df["Tenure"] = display_df["Tenure"].apply(
                        lambda x: f"{x:.0f} yrs" if pd.notna(x) and x != 0 else ""
                    )
        if "Maturity" in display_df.columns:
                    display_df["Maturity"] = display_df["Maturity"].apply(
                        lambda x: f"{x:.0f} yrs" if pd.notna(x) and x != 0 else ""
                    )

                # Custom styling function
            #def style_telegram_dataframe(df):
                    # Create a styler object
                    #styler = df.style
                    
                    # Highlight high potential customers
            #if "Potential_Level" in df.columns:
            #            styler = styler.apply(
            #                lambda row: ["background-color: #ff9999" if str(row.get("Potential_Level", "")).strip().upper() == "H" else "" for _ in row], 
            #                axis=1
            #           )
                    
                    # Color code based on potential
            #    def color_potential(val):
            #            if str(val).strip().upper() == "H":
            #                return "color: #d32f2f; font-weight: bold;"  # Red for High
            #            elif str(val).strip().upper() == "M":
            #                return "color: #f57c00; font-weight: bold;"  # Orange for Medium
            #            elif str(val).strip().upper() == "L":
            #                return "color: #388e3c; font-weight: bold;"  # Green for Low
            #            return ""
                    
            #    if "Potential_Level" in df.columns:
            #            styler = styler.map(color_potential, subset=["Potential_Level"])
                    
                    # Set properties for better display
            #        styler = styler.set_properties(**{
            #           'text-align': 'left',
            #            'white-space': 'pre-wrap',
            #            'font-size': '14px'
            #        })
                    
                    # Set table headers style
            #        styler = styler.set_table_styles([{
            #            'selector': 'th',
            #            'props': [('background-color', '#2E8B57'), 
            #                    ('color', 'white'),
            #                    ('font-weight', 'bold'),
            #                    ('text-align', 'center')]
            #        }])
                    
             #       return styler

        st.subheader("👥 Customer Visit Data")
                # Statistics
        total_visits = len(telegram_df)
        high_potential = len(
                    telegram_df[telegram_df["Potential_Level"].str.strip().str.upper() == "H"]
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Visits", total_visits)
        col2.metric("Total HC", 4)
        col3.metric("High Potential", high_potential)
        col4.metric(
                    "HP Percentage",
                    (
                        f"{(high_potential/total_visits*100):.1f}%"
                        if total_visits
                        else "0%"
                    ),
        )
                
                # Display styled dataframe
        #styled_df = style_telegram_dataframe(display_df)
        st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=800,
            )
                # Download option
        csv = telegram_df.to_csv(index=False)
        st.download_button(
                    label="📥 Download Visit Data",
                    data=csv,
                    file_name=f"customer_visits_{pres_start_date}_{pres_end_date}.csv",
                    mime="text/csv",
            )
    
    DB_NAME = "/Users/thekhemfee/Downloads/Customer_Network/CusXRealTime/customer_locations.db"

    with tab3:
        st.markdown(
            """
            <div class="function-card">
                <h2>🌍 Customer Analytics & Geographic Visualization</h2>
                <p>Analyze customer distribution and patterns on an interactive map</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Fetch data from CSV file
        try:
            telegram_df = pd.read_csv("cusinfo.csv")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            telegram_df = pd.DataFrame()

        if telegram_df.empty:
            st.info("No customer data available. Please ensure the CSV file exists and contains data.")
        else:
            # Display basic information about the data
            st.success(f"✅ Successfully loaded {len(telegram_df)} customer records")
            
            # Show basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(telegram_df))
            with col2:
                if 'latitude' in telegram_df.columns and 'longitude' in telegram_df.columns:
                    with_location = len(telegram_df.dropna(subset=['latitude', 'longitude']))
                    st.metric("With Location Data", with_location)
            with col3:
                if 'potential' in telegram_df.columns:
                    high_potential = len(telegram_df[telegram_df['potential'].str.strip().str.upper() == 'H'])
                    st.metric("High Potential", high_potential)

            # Map controls card
            with st.container():
                col1, col2 = st.columns([3,1])
                with col1:
                    show_map = st.checkbox("Display Interactive Map", value=True)
                with col2:
                    if show_map:
                        map_style = st.radio("Map Style", ["Street", "Satellite"], horizontal=True)

            if show_map:
                # Prepare data for mapping - check for different possible column names
                lat_col = next((col for col in telegram_df.columns if 'lat' in col.lower()), None)
                lon_col = next((col for col in telegram_df.columns if 'lon' in col.lower()), None)
                
                if lat_col and lon_col:
                    map_data = telegram_df.dropna(subset=[lat_col, lon_col]).copy()
                    
                    if not map_data.empty:
                        # Create map
                        avg_lat = map_data[lat_col].mean()
                        avg_lon = map_data[lon_col].mean()
                        
                        m = folium.Map(
                            location=[avg_lat, avg_lon], 
                            zoom_start=12,
                            tiles="OpenStreetMap" if map_style == "Street" else "Esri.WorldImagery"
                        )
                        
                        # Define color coding based on potential (if available)
                        POTENTIAL_COLORS = {
                            'H': 'red',      # High potential - Red
                            'M': 'orange',   # Medium potential - Orange
                            'L': 'green',    # Low potential - Green
                            '': 'blue'       # Unknown - Blue
                        }
                        
                        # Find potential column if exists
                        potential_col = next((col for col in telegram_df.columns if 'potential' in col.lower()), None)
                        # Find name column
                        name_col = next((col for col in telegram_df.columns if any(x in col.lower() for x in ['name', 'customer'])), 'Customer')
                        
                        # Add markers for each customer
                        for _, row in map_data.iterrows():
                            # Determine marker color based on potential
                            potential = str(row.get(potential_col, '')).strip().upper() if potential_col else ''
                            color = POTENTIAL_COLORS.get(potential, 'blue')
                            
                            # Create popup content with available data
                            popup_html = f"<div style='width: 250px; font-size: 12px;'><h4>{row.get(name_col, 'Unknown Customer')}</h4>"
                            
                            # Add available information to popup
                            for col in map_data.columns:
                                if col not in [lat_col, lon_col] and pd.notna(row.get(col)) and col != 'cluster':
                                    popup_html += f"<b>{col}:</b> {row[col]}<br>"
                            
                            popup_html += "</div>"
                            
                            # Add marker to map
                            folium.Marker(
                                location=[row[lat_col], row[lon_col]],
                                popup=folium.Popup(popup_html, max_width=300),
                                tooltip=row.get(name_col, 'Unknown Customer'),
                                icon=folium.Icon(color=color, icon='user', prefix='fa')
                            ).add_to(m)
                        
                        # Add heatmap layer
                        heat_data = [[row[lat_col], row[lon_col]] for _, row in map_data.iterrows()]
                        HeatMap(heat_data, radius=15).add_to(m)
                        
                        # Display the map
                        st_folium(m, width=1800, height=800)
                        
                        # Show map statistics
                        st.info(f"📍 Displaying {len(map_data)} customers with location data on the map")
                        
                    else:
                        st.warning("No location data available for mapping. Please ensure the CSV file contains latitude and longitude coordinates.")
                else:
                    st.warning("Could not find latitude/longitude columns in the data. Please check your CSV file structure.")

            # Additional analytics
            st.markdown('<div class="header-style">📊 Customer Insights</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Business type distribution
                biz_col = next((col for col in telegram_df.columns if any(x in col.lower() for x in ['biz', 'business', 'type'])), None)
                if biz_col and biz_col in telegram_df.columns:
                    biz_counts = telegram_df[biz_col].value_counts()
                    if len(biz_counts) > 0:
                        fig = px.pie(values=biz_counts.values, names=biz_counts.index, 
                                title="Business Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Potential distribution
                potential_col = next((col for col in telegram_df.columns if 'potential' in col.lower()), None)
                if potential_col and potential_col in telegram_df.columns:
                    potential_counts = telegram_df[potential_col].astype(str).str.strip().str.upper().value_counts()
                    if len(potential_counts) > 0:
                        fig = px.bar(x=potential_counts.index, y=potential_counts.values,
                                title="Customer Potential Distribution",
                                color=potential_counts.index,
                                color_discrete_map={'H': 'red', 'M': 'orange', 'L': 'green', 'NAN': 'gray'})
                        st.plotly_chart(fig, use_container_width=True)
            # Data preview
            with st.expander("📋 View Raw Data"):
                st.dataframe(telegram_df, use_container_width=True)
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d; margin-top: 30px;'>"
        "Sales Performance Dashboard • CMCB Bank • "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
