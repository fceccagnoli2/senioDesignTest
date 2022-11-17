import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
# sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from data import forecastingcodev2

##Trying to Put Logo in Top Right (Can use markdown)##
col1,col2,col3 = st.columns(3, gap = 'large')
# main_dir = st.session_state['directory']
steelcaseLogo = Image.open('assets/SteelcaseLogo.png')
with col3:
    st.image(steelcaseLogo, width = 100)

## Format Title ##
def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Product Line View
        </h1>

    """, unsafe_allow_html=True
                )
head()

# Load Data
data = st.session_state['data']
data_frame = data.clean_data

data_frame['late_bin'] = np.where(data_frame['days_late'] > 0, 1, 0)
# Section for I/O functionality
st.markdown("""
                <h4 style='text-align: center; margin-bottom: -35px;'>
                Upload CSV with list of components. Pleas input material numbers in column format. 
                </h4>

            """, unsafe_allow_html=True
            )
spectra = st.file_uploader("upload file", type={"csv", "txt"})
if spectra is not None:
    spectra_df = pd.read_csv(spectra)
    # Get Part Data
    parts = spectra_df.iloc[:,0].to_list()
    part_data = data_frame[data_frame['Material'].isin(parts)]
    # Columnate Their Input, Their Suppliers, and their Material Groups
    ic1, ic2 = st.columns(2, gap='small')
    with ic1:
        # Display Inputs
        st.write('Your Input:')
        st.write(spectra_df)
    with ic2:
        # Display Material Groups and Suppliers
        st.write('Suppliers and Material Groups')
        supp_info = part_data.groupby(['Material','Supplier','Material_Group']).size().reset_index()
        st.dataframe(pd.merge(
            spectra_df, supp_info, how='left', left_on=(spectra_df.columns[0]), right_on='Material')[
                     ['Material','Supplier','Material_Group']])
    # Display performance of Parts over time
    st.write('Input Material Performance vs Supply Chain as a Whole')
    # Columnate Page
    mc1, mc2, mc3 = st.columns(3, gap='small')
    mc1.metric(label='Average On-Time %', value=(1 - part_data['late_bin'].mean()),
                      delta=data_frame['late_bin'].mean() - part_data['late_bin'].mean())
    mc2.metric(label='Average Lead Time', value=part_data['expected_lead_time'].mean(),
                      delta=-data_frame['expected_lead_time'].mean() + part_data['expected_lead_time'].mean(),
                      delta_color='inverse')
    mc3.metric(label='Quantity of Parts Ordered', value=part_data['Invoice Qty'].sum(),
                      delta=-data_frame['Invoice Qty'].sum() + part_data['Invoice Qty'].sum())
    # Display Chart Graphic
    st.markdown("""
                    <h4 style='text-align: center; margin-bottom: -35px;'>
                    Late Delivery Percentage over Time. 
                    </h4>

                """, unsafe_allow_html=True
                )

    monthly_perf_viz = pd.DataFrame(part_data.resample('M', on='Scheduled_relevant_delivery_date').late_bin.mean())
    monthly_perf_viz = monthly_perf_viz.rename(columns={'late_bin': 'Percentage Late'})
    monthly_perf_viz["Percentage Late"] = monthly_perf_viz["Percentage Late"].rolling(4).mean()
    # include index data in image
    indexData = pd.DataFrame(
        [["07/2019", -0.46], ["08/2019", -0.33], ["09/2019", 0.11], ["10/2019", .03],
         ["11/2019", 0.1], ["12/2019", 0.01], ["01/2020", 0.04], ["02/2020", 1.1], ["03/2020", 2.46],
         ["04/2020", 3.10], ["05/2020", 2.55], ["06/2020", 2.30], ["07/2020", 2.80], ["08/2020", 1.33],
         ["09/2020", 0.60], ["10/2020", 0.11], ["11/2020", 0.7], ["12/2020", 1.64], ["01/2021", 1.42],
         ["02/2021", 1.90], ["03/2021", 2.18], ["04/2021", 2.52], ["05/2021", 2.97], ["06/2021", 2.70],
         ["07/2021", 2.94], ["08/2021", 3.23], ["09/2021", 3.24], ["10/2021", 3.79], ["11/2021", 4.22],
         ["12/2021", 4.30], ["01/2022", 3.64], ["02/2022", 2.76], ["03/2022", 2.78], ["04/2022", 3.42],
         ["05/2022", 2.62]],
        columns=["Date", "Value"])
    indexData["Date"] = pd.to_datetime(indexData["Date"])
    pl_fig = make_subplots(specs=[[{"secondary_y": True}]])
    pl_fig.add_scatter(x=monthly_perf_viz.index, y=monthly_perf_viz['Percentage Late'],fill='tonexty',
                       name="Performance of Materials in Product")
    pl_fig.add_scatter(x=indexData["Date"], y=[.01]*len(indexData), mode='lines', line=dict(
        color='#ff0000',
        width=4,
        dash='dashdot'
    ), name='Goal Performance'
                       )
    pl_fig.add_trace(go.Scatter(x=indexData["Date"], y=indexData["Value"],
                                 name="Global Supply Chain Pressure Index (GSCPI)"),
                      secondary_y=True)
    pl_fig.update_yaxes(title_text="Global Supply Chain Pressure Index (GSCPI)", secondary_y=True)
    pl_fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=12,
                color="black"
            ),
            bgcolor="LightBlue",
            bordercolor="Black",
            borderwidth=1
        )
    )
    pl_fig.layout.yaxis.tickformat = ',.0%'
    # fig.update_traces(line=dict(color="Black", width=0.5))
    st.plotly_chart(pl_fig, use_container_width=True)

    # Identify opportunities for growth
    st.markdown("""
                    <h4 style='text-align: center; margin-bottom: -35px;'>
                    Opportunities for Improvement
                    </h4>

                """, unsafe_allow_html=True
                )

    best_supplier_data = {}
    for mg in supp_info['Material_Group'].unique():
        score_data = data_frame[data_frame['Material_Group'] == mg]
        score_data = score_data.groupby('Supplier').agg('mean')[['late_bin', 'expected_lead_time']]
        standard = score_data.apply(
            lambda iterator: ((iterator.max() - iterator) / (iterator.max() - iterator.min())).round(2))
        score_data_rename = {
            'late_bin': 'Average Late %',
            'expected_lead_time': 'Average Lead Time',
        }
        standard_data_rename = {
            'late_bin': 'Standardized: Average Late %',
            'expected_lead_time': 'Standardized: Average Lead Time',
        }
        view_score_data = pd.merge(score_data.rename(columns=score_data_rename),
                                   standard.rename(columns=standard_data_rename),
                                   how='left', left_index=True, right_index=True)
        view_score_data['Score'] = (.6 * view_score_data['Standardized: Average Late %'] + \
                                   .4 * view_score_data['Standardized: Average Lead Time'])*100
        view_score_data = view_score_data.sort_values('Score', ascending=False)[['Score',
                                                                                 'Average Late %',
                                                                                 'Average Lead Time',
                                                                                 ]]
        best_supplier_data[mg] = view_score_data
    mat_selector = st.selectbox(
        'Material',
        supp_info['Material'].unique()
    )
    if mat_selector is not None:
        mg_selector = supp_info[supp_info['Material'] == mat_selector]['Material_Group'].reset_index()['Material_Group'][0]
        st.write(f"Showing the best suppliers for material: {mat_selector}")
        st.write(f"In material group: {mg_selector}")
        def highlight_max(s, threshold, column):
            is_max = pd.Series(data=False, index=s.index)
            is_max[column] = s.loc[column] >= threshold
            return ['background-color: green' if is_max.any() else '' for v in is_max]


        thresh = best_supplier_data[mg_selector]['Score'].max()
        view_score_data = best_supplier_data[mg_selector].reset_index(level=0)
        st.dataframe(
            view_score_data.style.apply(
                highlight_max, threshold=thresh, column=['Score'], axis=1).set_precision(2)
        )
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(view_score_data)
        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )
