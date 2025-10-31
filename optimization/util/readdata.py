import config as cfg
import pandas as pd

class DataReader:
    """
    A class to handle reading and processing data from CSV files.
    """

    @staticmethod
    def read_and_process(file_path):
        """
        Reads a CSV file, sets the 'Datetime' column as the index, and converts data to numeric.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        # Read the data from the CSV file
        df = pd.read_csv(file_path)

        # Set the 'Datetime' column as the index for the DataFrame
        df.set_index('Datetime', inplace=True)

        # Convert all data to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        return df

    @staticmethod
    def get_building_energy_demand():
        """
        Retrieves building energy demand data.

        Returns:
            np.ndarray: Flattened array of building energy demand values.
        """
        df = DataReader.read_and_process("../data/processed/building_data.csv")
        return df.values.flatten()

    @staticmethod
    def get_electricity_price():
        """
        Retrieves electricity price data for G2B and G2V scenarios.

        Returns:
            tuple: Flattened arrays of G2B and G2V electricity prices.
        """
        g2b_df = DataReader.read_and_process("../data/processed/electricitycostG2B_data.csv")
        g2v_df = DataReader.read_and_process("../data/processed/electricitycostG2V_data.csv")
        return g2b_df.values.flatten(), g2v_df.values.flatten()

    @staticmethod
    def get_photovoltaic_generation():
        """
        Calculates photovoltaic (PV) generation based on radiation and temperature data.

        Returns:
            list: PV generation values.
        """
        radiation_df = DataReader.read_and_process("../data/processed/radiation_data.csv")
        temperature_df = DataReader.read_and_process("../data/processed/temperature_data.csv")

        efficiency = 0.2

        # Calculate PV generation using the formula:
        # Efficiency * PV Area * Radiation (converted to kW/m^2) * Temperature correction factor
        pv_generation = [
            efficiency * cfg.PV_AREA * (rad * 0.2778) * (1 - (0.005 * (temp - 25)))
            for rad, temp in zip(radiation_df.values.flatten(), temperature_df.values.flatten())
        ]

        return pv_generation