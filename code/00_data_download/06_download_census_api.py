import os
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CENSUS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "census_acs")
os.makedirs(CENSUS_DIR, exist_ok=True)

# Variables we need to query
# B01003_001E: Total Population
# B19013_001E: Median Household Income
# B23025_002E: In labor force, B23025_004E: Employed
# B01001_002E: Male, B01001_026E: Female
# B03002_003E: White alone (not Hispanic), B03002_004E: Black/AA alone...
VARIABLES = {
    'NAME': 'Tract_Name',
    'B01003_001E': 'Total_Population',
    'B19013_001E': 'Median_Household_Income',
    'B23025_002E': 'In_Labor_Force',
    'B23025_004E': 'Employed',
    'B01001_002E': 'Male_Population',
    'B01001_026E': 'Female_Population',
    'B03002_003E': 'White_NonHispanic',
    'B03002_004E': 'Black_NonHispanic',
    'B03002_006E': 'Asian_NonHispanic',
    'B03002_012E': 'Hispanic_or_Latino'
}

def fetch_census_data(year):
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = ",".join(VARIABLES.keys())
    
    # State 06 is California, fetching for all tracts
    params = {
        'get': get_vars,
        'for': 'tract:*',
        'in': 'state:06'
    }
    
    logging.info(f"Fetching ACS 5-Year Data for {year}...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        headers = data[0]
        rows = data[1:]
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Rename columns to human-readable names
        df = df.rename(columns=VARIABLES)
        
        # Create FIPS code from state + county + tract
        df['FIPS'] = df['state'] + df['county'] + df['tract']
        
        # Save to CSV
        output_file = os.path.join(CENSUS_DIR, f"acs5_{year}_california_tracts.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"Saved {year} Census data to {output_file} shape: {df.shape}")
        return True
    else:
        logging.error(f"Failed to fetch {year} data. Status code: {response.status_code}")
        logging.error(response.text)
        return False

if __name__ == "__main__":
    fetch_census_data(2019)
    # The paper uses Jan 2021-June 2023 for period 2. ACS 2021 5-Year data fits best.
    fetch_census_data(2021)
