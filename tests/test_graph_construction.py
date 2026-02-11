"""
Unit tests for the graph construction module.
"""

import pytest
import pandas as pd
import networkx as nx
from src.data_processing import generate_synthetic_transactions
from src.graph_construction import TransactionGraph, compute_graph_metrics


class TestTransactionGraph:
    """Tests for TransactionGraph class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = generate_synthetic_transactions(
            n_accounts=20, n_transactions=100, random_state=42
        )
    
    def test_init(self):
        """Test initialization."""
        tg = TransactionGraph(directed=True)
        assert tg.directed is True
        assert isinstance(tg.graph, nx.DiGraph)
        
        tg_undirected = TransactionGraph(directed=False)
        assert isinstance(tg_undirected.graph, nx.Graph)
    
    def test_load_data(self):
        """Test data loading."""
        tg = TransactionGraph()
        tg.load_data(self.test_data)
        assert tg.transaction_data is not None
        assert len(tg.transaction_data) == len(self.test_data)
    
    def test_build_graph(self):
        """Test graph building."""
        tg = TransactionGraph(directed=True)
        tg.load_data(self.test_data)
        graph = tg.build()
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        assert isinstance(graph, nx.DiGraph)
    
    def test_graph_statistics(self):
        """Test graph statistics computation."""
        tg = TransactionGraph(directed=True)
        tg.load_data(self.test_data)
        tg.build()
        
        stats = tg.get_basic_stats()
        assert 'nodes' in stats
        assert 'edges' in stats
        assert 'density' in stats
        assert stats['nodes'] > 0
        assert stats['edges'] > 0
    
    def test_export_import_graph(self):
        """Test graph export and import."""
        tg = TransactionGraph(directed=True)
        tg.load_data(self.test_data)
        tg.build()
        
        # Export
        export_path = '/tmp/test_graph.gexf'
        tg.export_graph(export_path, format='gexf')
        
        # Import
        tg_imported = TransactionGraph(directed=True)
        imported_graph = tg_imported.import_graph(export_path, format='gexf')
        
        # Basic check - node count should match
        assert imported_graph.number_of_nodes() == tg.graph.number_of_nodes()


class TestGraphMetrics:
    """Tests for graph metrics computation."""
    
    def test_compute_graph_metrics(self):
        """Test metrics computation."""
        # Create simple test graph
        graph = nx.DiGraph()
        graph.add_edges_from([
            ('A', 'B', {'weight': 100}),
            ('B', 'C', {'weight': 200}),
            ('C', 'A', {'weight': 150}),
        ])
        
        metrics = compute_graph_metrics(graph)
        
        assert 'basic' in metrics
        assert metrics['basic']['nodes'] == 3
        assert metrics['basic']['edges'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
