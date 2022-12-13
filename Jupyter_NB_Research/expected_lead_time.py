def avg_lead_time(text_file):
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(text_file, sep = ",")
    required_columns = data[['Supplier', 'Material', 'Material_Group', 'days_late', 'expected_lead_time']]
    on_time = required_columns[required_columns['days_late']<= 0]
    lead_time = on_time.groupby(['Material_Group', 'Supplier'])['expected_lead_time'].mean().reset_index()
    return lead_time

avg_lead_time("model_data3.txt")
