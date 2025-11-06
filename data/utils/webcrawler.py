import requests
# from bs4 import BeautifulSoup
import pandas as pd
import os
import sys  

sys.path.append(os.path.abspath(".."))

def get_building_data(start_date="2023-01-01 00:00:00", end_date="2024-09-30 23:00:00"):
    
    # Define the URL and the form data for submission
    url = "https://epower.ga.ntu.edu.tw/fn4/report2.aspx"
    all_dataframes = []

    dates = pd.date_range(start=start_date, end=end_date, freq="d")
    dates = dates.strftime('%Y-%m-%d')

    for date in dates:

        date_times = pd.date_range(start=date+" 00:00:00", end=date+" 23:00:00", freq="h")
        
        zone = "N3"
        
        raw_path = f'./raw/{zone}-BuildingData-{date}.csv'
        
        if os.path.isfile(raw_path):
            all_dataframes.append(pd.read_csv(raw_path))
            continue
        
        else:

            # Example form data for submission (replace these values with the desired ones)
            form_data = {
                "ctg": zone,  # Example: "總配電站"
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
                df['datetime'] = date_times
                
                df.to_csv(raw_path, index=True)
                
                all_dataframes.append(df)
                
            else:
                print(f"Failed to fetch the page. Status code: {response.status_code}")
            
    if all_dataframes:
        complete_df = pd.concat(all_dataframes, ignore_index=True)
                    
    return complete_df
    
    