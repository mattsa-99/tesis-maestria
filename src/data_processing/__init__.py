"""
Data Processing Module
Handles loading, cleaning, and preprocessing of financial transaction data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class TransactionDataProcessor:
    """
    Processes financial transaction data for graph construction and analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or {}
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load transaction data from file.
        
        Args:
            filepath: Path to the data file (CSV, Excel, or Parquet)
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            DataFrame with transaction data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            self.data = pd.read_csv(filepath, **kwargs)
        elif filepath.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(filepath, **kwargs)
        elif filepath.suffix == '.parquet':
            self.data = pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        return self.data
    
    def clean_data(self, remove_duplicates: bool = True, 
                   handle_missing: str = 'drop') -> pd.DataFrame:
        """
        Clean the transaction data.
        
        Args:
            remove_duplicates: Whether to remove duplicate transactions
            handle_missing: How to handle missing values ('drop', 'fill', or 'keep')
            
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        cleaned = self.data.copy()
        
        # Remove duplicates if requested
        if remove_duplicates:
            cleaned = cleaned.drop_duplicates()
        
        # Handle missing values
        if handle_missing == 'drop':
            cleaned = cleaned.dropna()
        elif handle_missing == 'fill':
            # Fill numeric columns with median, categorical with mode
            for col in cleaned.columns:
                if cleaned[col].dtype in ['float64', 'int64']:
                    cleaned[col].fillna(cleaned[col].median(), inplace=True)
                else:
                    cleaned[col].fillna(cleaned[col].mode()[0] if len(cleaned[col].mode()) > 0 else 'Unknown', inplace=True)
        
        self.processed_data = cleaned
        return cleaned
    
    def validate_schema(self, required_columns: list) -> bool:
        """
        Validate that the data has required columns.
        
        Args:
            required_columns: List of required column names
            
        Returns:
            True if all required columns are present
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        missing_cols = set(required_columns) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    def extract_transactions(self, 
                           source_col: str = 'source',
                           target_col: str = 'target', 
                           amount_col: str = 'amount',
                           timestamp_col: Optional[str] = 'timestamp') -> pd.DataFrame:
        """
        Extract and format transaction data for graph construction.
        
        Args:
            source_col: Column name for source account/entity
            target_col: Column name for target account/entity
            amount_col: Column name for transaction amount
            timestamp_col: Column name for timestamp (optional)
            
        Returns:
            DataFrame with standardized transaction format
        """
        data_to_process = self.processed_data if self.processed_data is not None else self.data
        
        if data_to_process is None:
            raise ValueError("No data available.")
        
        # Extract relevant columns
        required_cols = [source_col, target_col, amount_col]
        if timestamp_col and timestamp_col in data_to_process.columns:
            required_cols.append(timestamp_col)
        
        transactions = data_to_process[required_cols].copy()
        
        # Rename to standard names
        rename_dict = {
            source_col: 'source',
            target_col: 'target',
            amount_col: 'amount'
        }
        if timestamp_col and timestamp_col in transactions.columns:
            rename_dict[timestamp_col] = 'timestamp'
            # Convert to datetime
            transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
        
        transactions = transactions.rename(columns=rename_dict)
        
        return transactions
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute basic statistics about the transaction data.
        
        Returns:
            Dictionary with statistics
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        stats = {
            'total_transactions': len(self.data),
            'unique_accounts': len(set(self.data.get('source', [])) | set(self.data.get('target', []))),
            'date_range': None,
            'total_volume': None
        }
        
        if 'amount' in self.data.columns:
            stats['total_volume'] = self.data['amount'].sum()
            stats['mean_amount'] = self.data['amount'].mean()
            stats['median_amount'] = self.data['amount'].median()
        
        if 'timestamp' in self.data.columns:
            stats['date_range'] = (
                self.data['timestamp'].min(), 
                self.data['timestamp'].max()
            )
        
        return stats


def generate_synthetic_transactions(n_accounts: int = 100,
                                   n_transactions: int = 1000,
                                   fraud_ratio: float = 0.05,
                                   random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic financial transaction data for testing.
    
    Args:
        n_accounts: Number of unique accounts
        n_transactions: Number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic transaction data
    """
    np.random.seed(random_state)
    
    # Generate account IDs
    accounts = [f"ACC{i:05d}" for i in range(n_accounts)]
    
    # Generate transactions
    transactions = []
    n_fraud = int(n_transactions * fraud_ratio)
    
    for i in range(n_transactions):
        is_fraud = i < n_fraud
        
        # Normal transactions
        source = np.random.choice(accounts)
        target = np.random.choice([acc for acc in accounts if acc != source])
        
        # Amount distribution (log-normal for realism)
        if is_fraud:
            # Fraudulent transactions tend to be larger and follow unusual patterns
            amount = np.random.lognormal(mean=8, sigma=1.5)
        else:
            amount = np.random.lognormal(mean=5, sigma=1)
        
        # Timestamp (last 365 days)
        days_ago = np.random.randint(0, 365)
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
        
        transactions.append({
            'transaction_id': f"TXN{i:06d}",
            'source': source,
            'target': target,
            'amount': round(amount, 2),
            'timestamp': timestamp,
            'is_fraud': is_fraud
        })
    
    df = pd.DataFrame(transactions)
    return df.sort_values('timestamp').reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    processor = TransactionDataProcessor()
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_transactions()
    print(f"Generated {len(synthetic_data)} synthetic transactions")
    print(f"Fraud ratio: {synthetic_data['is_fraud'].mean():.2%}")
    print("\nSample transactions:")
    print(synthetic_data.head())
