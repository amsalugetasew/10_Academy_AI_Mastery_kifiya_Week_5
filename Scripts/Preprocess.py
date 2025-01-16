from transformers import AutoTokenizer
import tensorflow as tf
import pandas as pd
import logging
import sqlite3
import re
class Preprocess:
    def __init__(self):
        self.df = {}
        # Load a transformer tokenizer (e.g., bert-base-multilingual-cased or AfroXLMR)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    

    def preprocess_text(self, text):
        """
        Preprocess Amharic text using normalization and transformer-based tokenization.
        """
        if not text:
            return None
        
        # Normalize: Remove special characters and retain Amharic text
        text = re.sub(r'[^\u1200-\u137F\s]', '', text)  # Keep Amharic characters
        text = text.strip().lower()  # Lowercase and strip whitespace

        # Tokenize with the transformer tokenizer
        tokenized = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="tf"  # TensorFlow tensors
        )
        
        # Convert tensors to numpy arrays and then to lists
        tokenized = {key: value.numpy().tolist() if isinstance(value, tf.Tensor) else value for key, value in tokenized.items()}
        
        return tokenized

    # df is my dataframe with a column 'Message' containing the text
    # Add new columns for tokenization outputs
    def tokenize_dataframe(self, df, message_column='message'):
        """
        Tokenize the messages in a DataFrame and add tokenization outputs as new columns.
        """
        input_ids = []
        attention_masks = []
        token_type_ids = []

        for text in df[message_column]:
            if pd.notnull(text):  # Skip null messages
                tokenized_output = self.preprocess_text(text)
                if tokenized_output:
                    input_ids.append(tokenized_output['input_ids'])  # Convert tensor to list
                    attention_masks.append(tokenized_output['attention_mask'])
                    token_type_ids.append(tokenized_output.get('token_type_ids', [[None]]))
                else:
                    input_ids.append(None)
                    attention_masks.append(None)
                    token_type_ids.append(None)
            else:
                input_ids.append(None)
                attention_masks.append(None)
                token_type_ids.append(None)

        # Add tokenization outputs as new columns
        df['input_ids'] = input_ids
        df['attention_mask'] = attention_masks
        df['token_type_ids'] = token_type_ids
        return df
    
    def clean_structure(self, df):
        # Structure messages into a DataFrame
        structured_data = []
        #`df` is a DataFrame, convert it to a list of dictionaries
        if isinstance(df, pd.DataFrame):
            df = df.to_dict(orient='records')
    #     'Channel Title', 'Channel Username', 'ID', 'Message', 'Date',
    #    'Media Path', 'input_ids', 'attention_mask', 'token_type_ids'
        for message in df:
            structured_data.append({
                'Channel Title': message['Channel Title'],
                'Channel Username': message['Channel Username'],
                'ID': message['ID'],
                'Date': message['Date'],
                'Media Path': message['Media Path'],
                'Content': message.get('Message', ''),
                'input_ids': message.get('input_ids', []),
                'attention_mask': message.get('attention_mask', []),
                'token_type_ids': message.get('token_type_ids', [])
            })

        # Convert to DataFrame
        df = pd.DataFrame(structured_data)

        # Save structured data for further analysis
        # df.to_csv('structured_telegram_data.csv', index=False, encoding='utf-8')
        return df

    def store_preprocessed_data(self, df):
        """
        Store preprocessed data into a SQLite database, ensuring compatibility with SQLite data types.
        """
        # Replace NaN with None for SQLite compatibility
        df = df.where(pd.notnull(df), None)

        # Convert necessary columns to strings explicitly
        for column in ['Media Path', 'Content', 'input_ids', 'attention_mask', 'token_type_ids']:
            if column in df.columns:
                df[column] = df[column].astype(str)

        # Convert Date to ISO-8601 string format for SQLite
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Create a SQLite database
        conn = sqlite3.connect('telegram_data.db')
        cursor = conn.cursor()

        # Create a table for storing the data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS telegram_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            MESSAGE_ID INTEGER,
            Channel_Title TEXT,
            Channel_Username TEXT,
            Date TEXT,
            Media_Path TEXT,
            Content TEXT,
            input_ids TEXT,
            attention_mask TEXT,
            token_type_ids TEXT
        )
        ''')

        # Insert data into the database
        for _, row in df.iterrows():
            cursor.execute('''
            INSERT INTO telegram_messages (MESSAGE_ID, Channel_Title, Channel_Username, Date, Media_Path, Content, input_ids, attention_mask, token_type_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['ID'],
                row['Channel Title'],
                row['Channel Username'],
                row['Date'],
                row['Media Path'],
                row['Content'],
                row['input_ids'],
                row['attention_mask'],
                row['token_type_ids']
            ))

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    def ReadSavedDate(self, db_name, table_name):
        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)  # Replace with the path to your database file

        # Define the SQL query
        query = f'''
        SELECT * FROM {table_name}
        '''

        # Execute the query and load data into a Pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Close the connection
        conn.close()

        return df