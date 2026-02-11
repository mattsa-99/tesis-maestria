"""
Unit tests for the data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    TransactionDataProcessor,
    generate_synthetic_transactions
)


class TestTransactionDataProcessor:
    """Tests for TransactionDataProcessor class."""
    
    def test_init(self):
        """Test initialization."""
        processor = TransactionDataProcessor()
        assert processor.data is None
        assert processor.processed_data is None
        assert isinstance(processor.config, dict)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_transactions(
            n_accounts=50,
            n_transactions=200,
            fraud_ratio=0.1,
            random_state=42
        )
        
        assert len(data) == 200
        assert 'source' in data.columns
        assert 'target' in data.columns
        assert 'amount' in data.columns
        assert 'timestamp' in data.columns
        assert 'is_fraud' in data.columns
        
        # Check fraud ratio
        fraud_ratio = data['is_fraud'].mean()
        assert 0.08 <= fraud_ratio <= 0.12  # Allow some variance
    
    def test_load_and_clean_data(self):
        """Test loading and cleaning data."""
        # Generate test data
        test_data = generate_synthetic_transactions(n_accounts=10, n_transactions=50)
        
        # Save to CSV
        test_path = '/tmp/test_transactions.csv'
        test_data.to_csv(test_path, index=False)
        
        # Load and clean
        processor = TransactionDataProcessor()
        loaded_data = processor.load_data(test_path)
        cleaned_data = processor.clean_data()
        
        assert loaded_data is not None
        assert len(cleaned_data) > 0
        assert cleaned_data.isna().sum().sum() == 0  # No missing values after cleaning
    
    def test_extract_transactions(self):
        """Test transaction extraction."""
        processor = TransactionDataProcessor()
        test_data = generate_synthetic_transactions(n_accounts=10, n_transactions=50)
        processor.data = test_data
        
        transactions = processor.extract_transactions()
        
        assert 'source' in transactions.columns
        assert 'target' in transactions.columns
        assert 'amount' in transactions.columns
        assert len(transactions) == len(test_data)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        data1 = generate_synthetic_transactions(
            n_accounts=20, n_transactions=100, random_state=42
        )
        data2 = generate_synthetic_transactions(
            n_accounts=20, n_transactions=100, random_state=42
        )
        
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_validity(self):
        """Test that generated data is valid."""
        data = generate_synthetic_transactions(n_accounts=30, n_transactions=100)
        
        # Check no self-loops
        assert (data['source'] != data['target']).all()
        
        # Check amounts are positive
        assert (data['amount'] > 0).all()
        
        # Check timestamps are valid
        assert pd.to_datetime(data['timestamp']).notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
