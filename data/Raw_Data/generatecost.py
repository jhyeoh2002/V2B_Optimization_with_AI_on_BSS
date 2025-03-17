import numpy as np
import pandas as pd
from datetime import datetime

def generate_building_cost(chargetype, chosen_date):

    time_range = pd.date_range("00:00", "23:00", freq="h").time
    
    # Two-stage charging, high-voltage building
    twohigh_weekday_summer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("09:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("09:00").time())(time_range)],
                                [2.03, 5.05])
    twohigh_weekday_nonsummer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("06:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("06:00").time() <= x < pd.Timestamp("11:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("11:00").time() <= x < pd.Timestamp("14:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("14:00").time())(time_range)],
                                [1.85, 4.77, 1.85, 4.77])
    twohigh_sat_summer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("09:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("09:00").time())(time_range)],
                                [2.03, 2.18])
    twohigh_sat_nonsummer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("06:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("06:00").time() <= x < pd.Timestamp("11:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("11:00").time() <= x < pd.Timestamp("14:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("14:00").time())(time_range)],
                                [1.85, 2, 1.85, 2])
    twohigh_sunday_summer = np.piecewise(time_range, 
                                    [np.vectorize(lambda x: x >= pd.Timestamp("00:00").time())(time_range)],
                                [2.03])
    twohigh_sunday_nonsummer = np.piecewise(time_range, 
                                    [np.vectorize(lambda x: x >= pd.Timestamp("00:00").time())(time_range)],
                                [1.85])

    # Three-stage charging, high-voltage building
    threehigh_weekday_summer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("09:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("09:00").time() <= x < pd.Timestamp("16:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("16:00").time() <= x < pd.Timestamp("22:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("22:00").time())(time_range)],
                                [1.91, 4.39, 7.03, 4.39])
    threehigh_weekday_nonsummer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("06:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("06:00").time() <= x < pd.Timestamp("11:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("11:00").time() <= x < pd.Timestamp("14:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("14:00").time())(time_range)],
                                [1.75, 4.11, 1.75, 4.11])
    threehigh_sat_summer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("09:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("09:00").time())(time_range)],
                                [1.91, 2.04])
    threehigh_sat_nonsummer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("06:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("06:00").time() <= x < pd.Timestamp("11:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("11:00").time() <= x < pd.Timestamp("14:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("14:00").time())(time_range)],
                                [1.75, 1.89, 1.75, 1.89])
    threehigh_sunday_summer = np.piecewise(time_range, 
                                    [np.vectorize(lambda x: x >= pd.Timestamp("00:00").time())(time_range)],
                                [1.91])
    threehigh_sunday_nonsummer = np.piecewise(time_range, 
                                    [np.vectorize(lambda x: x >= pd.Timestamp("00:00").time())(time_range)],
                                [1.75])
    
    # Check day and season
    day = check_day(chosen_date)
    season = check_season(chosen_date)

    # Return
    if chargetype=='twohigh' and day=='weekday' and season=='summer':
        return list(twohigh_weekday_summer)
    elif chargetype=='twohigh' and day=='weekday' and season=='nonsummer':
        return list(twohigh_weekday_nonsummer)
    elif chargetype=='twohigh' and day=='sat' and season=='summer':
        return list(twohigh_sat_summer)
    elif chargetype=='twohigh' and day=='sat' and season=='nonsummer':
        return list(twohigh_sat_nonsummer)
    elif chargetype=='twohigh' and day=='sunday' and season=='summer':
        return list(twohigh_sunday_summer)
    elif chargetype=='twohigh' and day=='sunday' and season=='nonsummer':
        return list(twohigh_sunday_nonsummer)
    elif chargetype=='threehigh' and day=='weekday' and season=='summer':
        return list(threehigh_weekday_summer)
    elif chargetype=='threehigh' and day=='weekday' and season=='nonsummer':
        return list(threehigh_weekday_nonsummer)
    elif chargetype=='threehigh' and day=='sat' and season=='summer':
        return list(threehigh_sat_summer)
    elif chargetype=='threehigh' and day=='sat' and season=='nonsummer':
        return list(threehigh_sat_nonsummer)
    elif chargetype=='threehigh' and day=='sunday' and season=='summer':
        return list(threehigh_sunday_summer)
    elif chargetype=='threehigh' and day=='sunday' and season=='nonsummer':
        return list(threehigh_sunday_nonsummer)

"""c_G2V_t"""

def check_day(chosen_date):
    date_obj = datetime.strptime(chosen_date, '%Y-%m-%d')
    if date_obj.weekday() < 5:  # Mon~Fri: 0~4
        return 'weekday'
    elif date_obj.weekday() == 5:
        return 'sat'
    else:
        return 'sunday'

def check_season(chosen_date):
    date_obj = datetime.strptime(chosen_date, '%Y-%m-%d')
    if datetime(date_obj.year, 5, 16) <= date_obj <= datetime(date_obj.year, 10, 15): # summer: 5/16-10/15
        return 'summer'
    else:
        return 'nonsummer'
    
def generate_ev_cost(chargetype, chosen_date):

    time_range = pd.date_range("00:00", "23:00", freq="h").time
    
    # Low voltage EVs
    evlow_weekday_summer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("16:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("16:00").time() <= x < pd.Timestamp("22:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("22:00").time())(time_range)],
                                [2.05, 8.35, 2.05])
    evlow_weekday_nonsummer = np.piecewise(time_range, 
                                [np.vectorize(lambda x: x < pd.Timestamp("15:00").time())(time_range),
                                    np.vectorize(lambda x: pd.Timestamp("15:00").time() <= x < pd.Timestamp("21:00").time())(time_range),
                                    np.vectorize(lambda x: x >= pd.Timestamp("21:00").time())(time_range)],
                                [1.95, 8.13, 1.95])
    evlow_sat_summer = np.piecewise(time_range, 
                                    [np.vectorize(lambda x: x >= pd.Timestamp("00:00").time())(time_range)],
                                [2.05])
    evlow_sat_nonsummer = np.piecewise(time_range, 
                                    [np.vectorize(lambda x: x >= pd.Timestamp("00:00").time())(time_range)],
                                [1.95])
    evlow_sunday_summer = evlow_sat_summer
    evlow_sunday_nonsummer = evlow_sat_nonsummer

    # High voltage EVs
    evhigh_weekday_summer = 0.95 * evlow_weekday_summer
    evhigh_weekday_nonsummer = 0.95 * evlow_weekday_nonsummer
    evhigh_sat_summer = 0.95 * evlow_sat_summer
    evhigh_sat_nonsummer = 0.95 * evlow_sat_nonsummer
    evhigh_sunday_summer = 0.95 * evlow_sunday_summer
    evhigh_sunday_nonsummer = 0.95 * evlow_sunday_nonsummer

    # Check day and season
    day = check_day(chosen_date)
    season = check_season(chosen_date)
    
    # Return
    if chargetype=='evlow' and day=='weekday' and season=='summer':
        return list(evlow_weekday_summer)
    elif chargetype=='evlow' and day=='weekday' and season=='nonsummer':
        return list(evlow_weekday_nonsummer)
    elif chargetype=='evlow' and day=='sat' and season=='summer':
        return list(evlow_sat_summer)
    elif chargetype=='evlow' and day=='sat' and season=='nonsummer':
        return list(evlow_sat_nonsummer)
    elif chargetype=='evlow' and day=='sunday' and season=='summer':
        return list(evlow_sunday_summer)
    elif chargetype=='evlow' and day=='sunday' and season=='nonsummer':
        return list(evlow_sunday_nonsummer)
    elif chargetype=='evhigh' and day=='weekday' and season=='summer':
        return list(evhigh_weekday_summer)
    elif chargetype=='evhigh' and day=='weekday' and season=='nonsummer':
        return list(evhigh_weekday_nonsummer)
    elif chargetype=='evhigh' and day=='sat' and season=='summer':
        return list(evhigh_sat_summer)
    elif chargetype=='evhigh' and day=='sat' and season=='nonsummer':
        return list(evhigh_sat_nonsummer)
    elif chargetype=='evhigh' and day=='sunday' and season=='summer':
        return list(evhigh_sunday_summer)
    elif chargetype=='evhigh' and day=='sunday' and season=='nonsummer':
        return list(evhigh_sunday_nonsummer)
    