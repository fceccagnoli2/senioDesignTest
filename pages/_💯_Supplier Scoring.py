import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data import forecastingcodev2
import datetime
import gui


cmap = 'coolwarm'

# Configuration
st.set_page_config(page_icon="💯", page_title="Supplier Scoring", layout="wide")


#Magnify function testing
def magnify():
    return [dict(selector="th", props=[("font-size", "7pt")]),
            dict(selector="td", props=[('padding', "0em 0em")]),
            dict(selector="th:hover", props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'), ('font-size', '12pt')])
            ]


# Trying to Put Logo in Top Right (Can use markdown)
col1, col2, col3 = st.columns(3, gap='large')
# main_dir = st.session_state['directory']
steelcaseLogo = Image.open('assets/SteelcaseLogo.png')
with col3:
    st.image(steelcaseLogo, width=100)

# Page title
gui.colored_header(label='Supplier Scoring',
                   description='Assign suppliers in a particular material group scores based off of their past performance.',
                   color_name='blue-60')

st.write('#')
# Load Data
data = st.session_state['data']

# Write subheader
st.markdown("""
                <h4 style='text-align: center; margin-bottom: -35px;'>
                Select Material Group and Dates
                </h4>

            """, unsafe_allow_html=True
            )

# Create select box to choose material group
mat_grp_ranks = data.data['Material_Group'].value_counts()
mgs = mat_grp_ranks.reset_index()
mat_group1 = st.selectbox('Material Group Scores', mgs)
st.write('The current material group is ', mat_group1)

# Create start and end date filter
in_col11, in_col21 = st.columns(2, gap='small')
today = st.session_state['today']
with in_col11:
    start_date1 = st.date_input('Start Date', datetime.date(2019,1,1))
with in_col21:
    end_date1 = st.date_input('End Date', today)

# Display scoring formula
st.markdown("""
                <h6 style='text-align: center; margin-bottom: -35px;'>
                Scoring Formula
                </h6>
            """, unsafe_allow_html = True)
st.latex('''
        \small{Score = Weight*AvgLatePerc + Weight*AvgLeadTime + Weight * PriceChangeHeuristic}
        ''')

#Allow user input for scoring algorithm
scoring_col1, scoring_col2, scoring_col3 = st.columns(3, gap = 'small')
with scoring_col1:
    avg_late_weight = st.text_input('Weight for Average Late %', '60')
with scoring_col2:
    avg_lead_weight = st.text_input('Weight for Average Lead Time', '20')
with scoring_col3:
    price_heuristic_weight = st.text_input('Weight for Price Change Heuristic', '20')

submitted1 = st.checkbox('Find Best Supplier', value = False)

#Show the user the best suppliers for a given material group
if submitted1:
    # Get data
    fullData = data.SplitData(start_date1, end_date1)
    fullData['late_bin'] = np.where(fullData['days_late'] > 0, 1, 0)
    st.write("#")
    st.markdown("""
                    <h2 style='text-align: center; margin-bottom: -35px;'>
                    The Best Supplier is...
                    </h2>

                """, unsafe_allow_html=True
                )
    score_data = data.SplitData(start_date1, end_date1)

    score_data = score_data[score_data['Material_Group'] == mat_group1]
    score_data['late_bin'] = np.where(score_data['days_late'] > 0, 1, 0)
    score_data['costPerIndividualMaterial'] = score_data["Invoice ($)"] / score_data["Invoice Qty"]
    score_data['costPerIndividualMaterial'] = score_data['costPerIndividualMaterial'].round(2)
    cols = ["Material", "Invoice ($)", "Invoice Qty", "costPerIndividualMaterial",
            "Scheduled_relevant_delivery_date"]
    suppliers = score_data["Supplier"].drop_duplicates()

    priceData = []

    # Create price heuristic for each supplier
    for supplier in suppliers:
        heuristicsPerMaterial = []
        numOrders = []
        numPriceChanges = []
        supplierData = score_data[score_data["Supplier"] == supplier]
        materials = supplierData["Material"].drop_duplicates()
        for material in materials:
            a = score_data[(score_data["Material"] == material) & (score_data["Supplier"] == supplier)]
            prices = a[["costPerIndividualMaterial"]].drop_duplicates().dropna()
            prices = prices[prices["costPerIndividualMaterial"] > 0]
            if len(prices) - 1 > 0:
                numPriceChanges.append(len(prices) - 1)
                numOrdersAtPrice = []
                for price in prices["costPerIndividualMaterial"]:
                    k = a[a["costPerIndividualMaterial"] == price]
                    numOrdersAtPrice.append(len(k))

                prices["priceChange"] = prices["costPerIndividualMaterial"] - prices["costPerIndividualMaterial"].shift(
                    1)
                prices["numOrdersAtPrice"] = numOrdersAtPrice
                prices["heurisitc"] = np.where(prices["priceChange"] < 0,
                                                prices["priceChange"] * prices["numOrdersAtPrice"] * -1,
                                                prices["priceChange"] * prices["numOrdersAtPrice"])
                heurisitc = prices["heurisitc"].mean()
                heuristicsPerMaterial.append(heurisitc)
                numOrders.append(len(a))
        priceData.append([supplier, np.mean(heuristicsPerMaterial), sum(numOrders), sum(numPriceChanges)])

    priceData = pd.DataFrame(priceData, columns=["Supplier", "Heuristic", "numOrders", "numPriceChanges"])
    priceData["Price Change Heurisitc"] = priceData["Heuristic"] / priceData["numOrders"] * 100

    # Create dataframe for score data
    score_data = score_data.groupby('Supplier').agg({'Vendor_number': 'first', 'late_bin': 'mean', 'expected_lead_time': 'mean'})
    score_data = score_data.merge(priceData[["Supplier", "Price Change Heurisitc"]], how = 'left', on = "Supplier")
    score_data.index = score_data.pop("Supplier")
    standard = score_data.apply(
        lambda iterator: ((iterator.max() - iterator) / (iterator.max() - iterator.min())).round(2))

    score_data_rename = {
        'Vendor_number': 'Supplier #',
        'late_bin': 'Average Late %',
        'expected_lead_time': 'Average Lead Time'
    }
    standard_data_rename = {
        'late_bin': 'Standardized: Average Late %',
        'expected_lead_time': 'Standardized: Average Lead Time',
        'Price Change Heurisitc': "Standardized: Price Change Heurisitc"
    }

    # Finish up making the dataframe

    view_score_data = pd.merge(score_data.rename(columns=score_data_rename),
                                standard.rename(columns=standard_data_rename),
                                how='left', left_index=True, right_index=True)

    view_score_data['Score'] = float(avg_late_weight)/100 * view_score_data['Standardized: Average Late %'] + \
                                float(avg_lead_weight)/100 * view_score_data['Standardized: Average Lead Time'] + \
                                float(price_heuristic_weight)/100 * view_score_data['Standardized: Price Change Heurisitc']
    view_score_data = view_score_data.sort_values('Score', ascending=False)[['Supplier #',
                                                                            'Score',
                                                                            'Average Late %',
                                                                            # 'Standardized: Average Late %',
                                                                            'Price Change Heurisitc',
                                                                            # 'Standardized: Price Change Heurisitc',
                                                                            'Average Lead Time',
                                                                            # 'Standardized: Average Lead Time',
                                                                            ]]
    fake1, view_col, fake2 = st.columns([3, 7, 1])

    with view_col:
        # Highlight best supplier in green
        def highlight_max(s, threshold, column):
            is_max = pd.Series(data=False, index=s.index)
            is_max[column] = s.loc[column] >= threshold
            return ['background-color: green' if is_max.any() else '' for v in is_max]

        view_score_data = view_score_data.reset_index(level=0)
        view_score_data['Average Late %'] = view_score_data['Average Late %'] * 100
        view_score_data['Score'] = view_score_data['Score'] * 100
        thresh = view_score_data['Score'].max()
    if len(view_score_data) > 0:
        st.dataframe(
            view_score_data.style.apply(highlight_max, threshold=thresh, column=['Score'], axis=1).set_precision(2), use_container_width=True)
    else:
        st.error('There is no data for material group ' + mat_group1 + ' for this time frame. '
                'Please adjust the time or change the material group.', icon = '🚨')