# src/data_loader.py (OPTIMIZED)
"""
Data Loading and Preprocessing Module - Optimized for large datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and preprocess the airline dataset"""
    
    def __init__(self, data_path=None):
        if data_path is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            data_path = project_root / 'data' / 'Airline Dataset Updated - v2.csv'
        self.data_path = Path(data_path)
        self.df = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load the CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def get_data_info(self):
        """Get basic information about the dataset"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'null_counts': self.df.isnull().sum().to_dict(),
            'target_distribution': self.df['Flight Status'].value_counts().to_dict() if 'Flight Status' in self.df.columns else {}
        }
        return info
    
    def preprocess(self, target_col='Flight Status', drop_cols=None, sample_size=None):
        """
        Preprocess the data for decision tree algorithms
        
        Args:
            target_col: Name of the target column
            drop_cols: List of columns to drop
            sample_size: If provided, use a sample of the data for faster execution
        """
        if drop_cols is None:
            drop_cols = ['Passenger ID', 'First Name', 'Last Name', 'Pilot Name']
        
        # Make a copy
        df = self.df.copy()
        
        # Sample if specified (for faster execution)
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size} rows for faster execution")
        
        # Drop unnecessary columns
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Handle dates - extract features
        if 'Departure Date' in df.columns:
            df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')
            df['Departure Month'] = df['Departure Date'].dt.month.fillna(0).astype(int)
            df['Departure Day'] = df['Departure Date'].dt.day.fillna(0).astype(int)
            df = df.drop(columns=['Departure Date'])
        
        # Handle target variable
        self.target_col = target_col
        self.X_cols = [col for col in df.columns if col != target_col]
        
        # Count unique values per column for logging
        print("\nFeature unique value counts:")
        for col in self.X_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
        
        # Encode categorical variables
        for col in self.X_cols:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(df[target_col])
        X = df[self.X_cols].values
        
        return X, y, self.X_cols
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)