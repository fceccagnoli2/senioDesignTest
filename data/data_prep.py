# import
import pandas as pd

# Data ingestion class
class IngestionPipe:
    def __init__(self, features: list, target: str, dates: list = None, post_date=None,
                 pre_date=None):
        self.data = None
        self.clean_data = None
        self.features = features
        self.target = target
        self.dates = dates
        self.post_date = post_date
        self.pre_date = pre_date

    def dataAllocation(self, path: str):
        # TODO: Separate out the x_data and y_data and return each
        #  args: string path for .csv file
        #  return: None
        #  -------------------------------
        print("Beginning ingestion ========---------------------------------------------==========")
        all_data = pd.read_csv(path)
        if self.dates != None:
            for date_feild in self.dates:
                all_data[date_feild] = pd.to_datetime(all_data[date_feild], format='%Y-%m-%d')
        self.data = all_data
        print(
            f"Data size:\n  Number of Rows: {len(all_data)}\n  Number of Features: {len(self.features)}\n  Number of Targets: 1")
        # get rid of unwanted dates
        print(f"Cutting off date ranges:\n  Before: {self.post_date}\n  After: {self.pre_date}")
        split_index = (all_data['Scheduled_relevant_delivery_date'] <= self.post_date) & (
                    all_data['Scheduled_relevant_delivery_date'] >= self.pre_date)
        cleaner_data = all_data[split_index]
        # create clean data set
        self.clean_data = cleaner_data
        # -------------------------------
        print('Allocation complete ========---------------------------------------------==========\n')

    def SplitData(self, time_start: str, time_end: str):
        # TODO: get a specific window of time from the data
        #  for purposes such as plotting
        #  return: New DF
        #  -------------------------------
        print("Getting slice of data ========------------------------==========")
        time_start, time_end = pd.to_datetime(time_start, format='%Y-%m-%d'), pd.to_datetime(time_end, format='%Y-%m-%d')
        split_index = (self.clean_data['Scheduled_relevant_delivery_date'] >= time_start) & (
                self.clean_data['Scheduled_relevant_delivery_date'] < time_end)
        print('Data successfully sliced. ========-------------------------------------==========\n')
        return self.clean_data[split_index]


