# Arguments to use for data class
import pandas as pd
features = [
    'PO_Number',
    'PO_Line_Number',
    'Vendor_number',
    'Material',
    'Material_Group',
    'PO_Document_Date',
    'Scheduled_relevant_delivery_date',
    'Posting_Date',
    'days_late',
    'expected_lead_time',
    'material_order_history',
    'avg_material_late',
    'supplier_material_order_history',
    'avg_supplier_material_late',
    'supplier_total_order_history',
    'avg_supplier_general_late',
    'number_line_items',
    'avg_MOQ',
    'avg_unit_price',
    'avg_min_price',
    'delivery_month',
    'dayofweek',
    'time',
    'delivery_season'
]
target = "days_late"
categorical = [
    'Material_Group',
    'delivery_month',
    'dayofweek',
    'delivery_season'
]
dates = [
    'Scheduled_relevant_delivery_date',
    'Posting_Date',
    'PO_Document_Date'
]
path= 'model_data.csv'
clean_end_date = pd.to_datetime("2022-05-01")
clean_start_date = pd.to_datetime("2019-01-01")