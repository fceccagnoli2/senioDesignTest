import streamlit as st
import pandas as pd
import numpy as np
import warnings
from stqdm import stqdm
from datetime import datetime, timedelta
import os, sys

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
st.session_state['directory'] = main_dir



############### UPLOAD RELIABILITY FILE ###############
reliability_upload = st.file_uploader("Upload Reliability Data Here")
if reliability_upload is not None:
    reliability = pd.read_excel(reliability_upload, names = ['Supplier', 'Vendor_Number',
                                                    'Purchasing_Organization', 'PO_Document_Date',
                                                    'PO_Number', 'PO_Line_Number', 'Material', 'Material_Group',
                                                    'MRP_Controller', 'Changed_On_Date', 'Confirmed_Delivery_Date',
                                                    'Creation_Date_of_Confirmation', 'Posting_Date',
                                                    'Item_Delivery_Date', 'Scheduled_relevant_delivery_date',
                                                    'Supply_Chain_Error_Reason_Code'])
    st.write(reliability.head())


    ####### Clean the reliability data #########

    dates_to_fix = ['PO_Document_Date','Changed_On_Date','Confirmed_Delivery_Date',
                    'Creation_Date_of_Confirmation','Posting_Date','Item_Delivery_Date',
                    'Scheduled_relevant_delivery_date']

    for i in dates_to_fix:
        if i == 'Scheduled_relevant_delivery_date':
            reliability[i] = pd.to_datetime(reliability[i].astype(str))
        elif i == 'PO_Document_Date':
            reliability[i] = pd.to_datetime(reliability[i].astype(str), format = '%Y%m%d')
        else:
            reliability[i] = reliability[i].apply(lambda x: str(x))
            reliability[i] = reliability[i].apply(lambda x: datetime(year=int(x[0:4]), month=int(x[4:6]), day=int(x[6:8])) if (x != 'nan' and x!= '0.0') else np.nan)

    # Create an empty dataframe of all the columns and the length of the columns

    clean_reliability = pd.DataFrame(dict(zip(reliability.columns, [[]*len(reliability.columns)])))

    # Create a list of tuples that stores all of the unique PO Number/PO Line pairs

    nums_lines = reliability[['PO_Number','PO_Line_Number']].sort_values('PO_Number')
    numbers_lines = set(list(zip(nums_lines['PO_Number'].to_list(), nums_lines['PO_Line_Number'].to_list())))


    # Now, for each pair of PO Number/PO Line, we will get the most updated information
    # for each respective column in the dataframe and append it to our new dataframe

    for item in stqdm(numbers_lines): #PO Numbers
        ponum = item[0]
        polin = item[1]
        check = reliability[(reliability['PO_Number']==ponum)&(reliability['PO_Line_Number']==polin)]
        #most recent changed on date
        last_changed = check.sort_values(by='Changed_On_Date', ascending=False).iloc[0]['Changed_On_Date']
        #most recent error code
        error_final = check.sort_values(by='Changed_On_Date', ascending=False).iloc[0]['Supply_Chain_Error_Reason_Code']
        #most recent due date
        last_due_date = check.sort_values(by='Scheduled_relevant_delivery_date', ascending=False).iloc[0]['Scheduled_relevant_delivery_date']
        #most recent arrival date
        arrival = check.sort_values(by='Posting_Date', ascending=False).iloc[0]['Posting_Date']
        last = check.groupby(['PO_Number','PO_Line_Number']).first()
        new_row = pd.DataFrame({
            'PO_Number': ponum,
            'PO_Line_Number': polin,
            'Supplier': last['Supplier'],
            'Vendor_number': last['Vendor_Number'],
            'Material': last['Material'],
            'Material_Group': last['Material_Group'],
            'PO_Document_Date': last['PO_Document_Date'],
            'Scheduled_relevant_delivery_date': last_due_date,
            'Posting_Date': arrival,
            'error_code': error_final,
        })
        clean_reliability = pd.concat([clean_reliability,new_row])


    # Remove unnecessary columns

    reliability_data = clean_reliability[list(new_row.columns)].reset_index()
    reliability_data = reliability_data.drop(columns=['index'], axis = 1)


    # Fix data types of columns to ensure the merge with te Cost Data goes smoothly

    cols_to_change = ['PO_Number','PO_Line_Number', 'Vendor_number']
    reliability_data[cols_to_change] = reliability_data[cols_to_change].applymap(np.int64)
    reliability_data['Vendor_number'] = reliability_data['Vendor_number'].astype(object)


    # st.write(reliability_data)






############### UPLOAD COST FILE ###############
cost_upload = st.file_uploader("Upload Cost Data Here")
if cost_upload is not None:
    cost = pd.read_excel(cost_upload, header=3,
                         names=['PO_Number', 'PO_Line_Number', 'Material', 'Material Description', 'Material_Group',
                                'Material Group Description', 'Vendor_number', 'Invoice ($)', 'Invoice Qty',
                                'Purchase Order $', 'Purchase Order Qty'])


    # Remove "NA/" from Vendor_number and Material columns
    cost[['Vendor_number', 'Material']] = cost[['Vendor_number', 'Material']].applymap(lambda x: x[3:])

    # Remove similar columns from Cost Data except for PO_Number and PO_Line_Number
    cost_data = cost.drop(columns=['Material', 'Vendor_number', 'Material_Group'])




############### MERGE THE TWO CLEANED FILES ###############
    merged_data = pd.merge(reliability_data, cost_data, how='left', left_on=('PO_Number', 'PO_Line_Number'),
                           right_on=('PO_Number', 'PO_Line_Number'))

    merged_data['days_late'] = pd.Series([(merged_data['Posting_Date'][pos] - merged_data['Scheduled_relevant_delivery_date'][pos]).days
         for pos in range(len(merged_data['Posting_Date']))])

    st.write(merged_data.head())

    data_path = main_dir + '/app/model_data4.csv'
    merged_data.to_csv(data_path)



