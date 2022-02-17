import pickle
import numpy as np
import pandas as pd

#---------------------------------------------------------This is a split line--
with open('df_raw.pickle','rb') as read_file:
    df_raw = pickle.load(read_file)
df = df_raw.copy()
# print(df.columns)
#---------------------------------------------------------This is a split line--
df = df.loc[:, 'ScheduledDay':'AppointmentDay']
# print(df.columns)
#---------------------------------------------------------This is a split line--
df['ScheduledDay_Day'] = df['ScheduledDay'].str.split(pat='T',
                                                      expand = True)[0]
df['ScheduledDay_Time'] = df['ScheduledDay'].str.split(pat='T',
                                                       expand = True)[1]
df['AppointmentDay_Day'] = df['AppointmentDay'].str.split(pat='T',
                                                          expand = True)[0]

df['AppointmentDay_Time'] = df['AppointmentDay'].str.split(pat='T',
                                                           expand = True)[1]
#---------------------------------------------------------This is a split line--
df.drop(['ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)
#---------------------------------------------------------This is a split line--
df['ScheduledDay_Time'] = df['ScheduledDay_Time'].str.translate({ord('Z'): None})
df['AppointmentDay_Time'] = df['AppointmentDay_Time'].str.translate({ord('Z'): None})
df['AppointmentDay_Time'] = np.nan
#---------------------------------------------------------This is a split line--
df['ScheduledDay_Hours'] = df['ScheduledDay_Time'].apply(lambda x:x[0:2])
#---------------------------------------------------------This is a split line--
df['ScheduledDay_Day'] = pd.to_datetime(df['ScheduledDay_Day'], infer_datetime_format=True)
df['AppointmentDay_Day'] = pd.to_datetime(df['AppointmentDay_Day'], infer_datetime_format=True)
#---------------------------------------------------------This is a split line--
df['ScheduledDay_Date'] = df['ScheduledDay_Day'].astype(str).apply(lambda x:x[-2:])
df['AppointmentDay_Date'] = df['AppointmentDay_Day'].astype(str).apply(lambda x:x[-2:])
df['day_difference'] = df['AppointmentDay_Day'] - df['ScheduledDay_Day']
df['day_difference'] = df['day_difference'].apply(lambda x: x.days)
#---------------------------------------------------------This is a split line--
# print(np.min(df['day_difference']))

# print(df['day_difference'])
#---------------------------------------------------------This is a split line--
print(df.__repr__())
