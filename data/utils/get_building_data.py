import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def get_building_data(startdate="2024-01-29",  enddate="2024-2-10"):
    
    if os.path.isfile('./Cleaned_Data/building.csv'):
        print("building data found.")
        
    else:
        # Define the URL and the form data for submission
        url = "https://epower.ga.ntu.edu.tw/fn4/report2.aspx"
        all_dataframes = []

        dates = pd.date_range(start=startdate, end=enddate, freq="d")
        dates = dates.strftime('%Y-%m-%d')

        for date in dates:

            date_times = pd.date_range(start=date+" 00:00:00", end=date+" 23:00:00", freq="h")

            # Example form data for submission (replace these values with the desired ones)
            form_data = {
                "ctg": "N3",  # Example: "總配電站"
                "dt1": date,  # Example date
                "ok": "確定",  # Submit button value
            }

            # Submit the form using a POST request
            response = requests.post(url, data=form_data)

            if response.status_code == 200:
                # Parse the response HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the table containing the data
                table = soup.find("table", {"border": "1", "class": "style3"})
                
                # Extract table rows
                rows = table.find_all("tr")
                
                # Extract header and data
                header = [th.text.strip() for th in rows[0].find_all("td")]
                data = [
                    [td.text.strip() for td in row.find_all("td")]
                    for row in rows[1:]
                ]

                # # Convert to a pandas DataFrame
                df = pd.DataFrame(data[2:-1], columns=data[0])
                # print(df)
                df['datetime'] = date_times
                
                all_dataframes.append(df)
                print(all_dataframes)
                
            else:
                print(f"Failed to fetch the page. Status code: {response.status_code}")
                
        if all_dataframes:
            complete_df = pd.concat(all_dataframes, ignore_index=True)
            
        complete_df.to_csv('./Raw_Data/all_building.csv', index=False)
        
        complete_df = pd.read_csv('./Raw_Data/all_building.csv')

        building = complete_df.drop(columns=['時間/館舍'])
        building['datetime'] = pd.to_datetime(building['datetime'])
        building = building.set_index('datetime')

        building['-推廣中心-'].to_csv('./Cleaned_Data/building.csv')

        building = pd.read_csv('./Cleaned_Data/building.csv')
        building = building.set_index('datetime')

        building['-推廣中心-'] = pd.to_numeric(building['-推廣中心-'], errors='coerce')
        building['filled'] = building['-推廣中心-'].fillna(building['-推廣中心-'].shift(-24 * 7))
        building['energy (kWh)'] = building['filled'].fillna(building['filled'].shift(24 * 7))
        building['energy (kWh)'].to_csv('./Cleaned_Data/building.csv')

        print(building)
        
        print("Done collecting data")
    
    