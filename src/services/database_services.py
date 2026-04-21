import pandas as pd
import yfinance as yf
import pandas as pd
import sqlite3
import os

from src.log.logs import LoggerHandler


class DatabaseService:
    def __init__(self):
        self.logger = LoggerHandler(__name__)
        self.list_symbol = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.start_date = '2002-01-01'
        self.end_date = '2026-04-20'

    def update_bronze_table(self):
        try:
            
            df = yf.download(
                self.list_symbol, 
                start=self.start_date, 
                end=self.end_date
            )

            df_index_ajusted = df.stack(level=1).reset_index()

            db_dir = "./src/database"

            os.makedirs(db_dir, exist_ok=True)

            db_path = os.path.abspath(os.path.join(db_dir, "bronze.db"))

            conn = sqlite3.connect(db_path)

            df_index_ajusted.to_sql("stocks_raw", conn, if_exists="replace", index=False)

            self.logger.INFO(f"Table 'stocks_raw' updated successfully in database.")
            
        except Exception as e:
            self.logger.ERROR(f"Error updating table 'stocks_raw': {e}")
            raise
        finally:
            conn.close()