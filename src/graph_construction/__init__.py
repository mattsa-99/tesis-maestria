"""
Graph Construction Module
Creates directed graphs from financial transaction data.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime


class TransactionGraph:
    """
    Constructs and manages directed graphs from financial transaction data.
    """
    
    def __init__(self, directed: bool = True):
        """
        Initialize the transaction graph.
        
        Args:
            directed: Whether to create a directed graph (default: True)
        """
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.transaction_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load transaction data.
        
        Args:
            data: DataFrame with columns: source, target, amount, [timestamp]
        """
        required_cols = ['source', 'target', 'amount']
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.transaction_data = data.copy()
        
    def build(self, 
             weight_col: str = 'amount',
             aggregate: str = 'sum',
             include_attributes: Optional[List[str]] = None) -> nx.Graph:
        """
        Build the graph from transaction data.
        
        Args:
            weight_col: Column to use for edge weights
            aggregate: How to aggregate multiple transactions ('sum', 'mean', 'count')
            include_attributes: Additional columns to include as edge attributes
            
        Returns:
            NetworkX graph
        """
        if self.transaction_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Clear existing graph
        self.graph.clear()
        
        # Group transactions by source-target pairs
        group_cols = ['source', 'target']
        
        if aggregate == 'sum':
            agg_dict = {weight_col: 'sum'}
        elif aggregate == 'mean':
            agg_dict = {weight_col: 'mean'}
        elif aggregate == 'count':
            agg_dict = {weight_col: 'count'}
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        # Add additional attributes if specified
        if include_attributes:
            for attr in include_attributes:
                if attr in self.transaction_data.columns:
                    agg_dict[attr] = 'first'
        
        grouped = self.transaction_data.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Add edges to graph
        for _, row in grouped.iterrows():
            edge_attrs = {
                'weight': float(row[weight_col]),
                'transaction_count': len(self.transaction_data[
                    (self.transaction_data['source'] == row['source']) &
                    (self.transaction_data['target'] == row['target'])
                ])
            }
            
            # Add additional attributes
            if include_attributes:
                for attr in include_attributes:
                    if attr in row:
                        edge_attrs[attr] = row[attr]
            
            self.graph.add_edge(row['source'], row['target'], **edge_attrs)
        
        # Store metadata
        self.metadata = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'is_directed': self.directed,
            'build_time': datetime.now(),
            'weight_col': weight_col,
            'aggregate': aggregate
        }
        
        return self.graph
    
    def add_node_attributes(self, attributes: Dict[str, Dict[str, Any]]) -> None:
        """
        Add attributes to nodes.
        
        Args:
            attributes: Dictionary mapping node IDs to attribute dictionaries
        """
        for node, attrs in attributes.items():
            if node in self.graph:
                nx.set_node_attributes(self.graph, {node: attrs})
    
    def get_subgraph(self, nodes: List[str]) -> nx.Graph:
        """
        Extract a subgraph containing specified nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Subgraph
        """
        return self.graph.subgraph(nodes).copy()
    
    def get_temporal_subgraph(self, 
                             start_time: datetime,
                             end_time: datetime) -> nx.Graph:
        """
        Extract subgraph for a specific time period.
        
        Args:
            start_time: Start of time period
            end_time: End of time period
            
        Returns:
            Temporal subgraph
        """
        if self.transaction_data is None or 'timestamp' not in self.transaction_data.columns:
            raise ValueError("No temporal data available.")
        
        # Filter transactions by time
        mask = (
            (self.transaction_data['timestamp'] >= start_time) &
            (self.transaction_data['timestamp'] <= end_time)
        )
        temporal_data = self.transaction_data[mask]
        
        # Create new graph
        temp_graph = TransactionGraph(directed=self.directed)
        temp_graph.load_data(temporal_data)
        temp_graph.build()
        
        return temp_graph.graph
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_directed': self.directed
        }
        
        if self.directed:
            stats['reciprocity'] = nx.reciprocity(self.graph)
        
        # Connected components
        if self.directed:
            stats['strongly_connected_components'] = nx.number_strongly_connected_components(self.graph)
            stats['weakly_connected_components'] = nx.number_weakly_connected_components(self.graph)
        else:
            stats['connected_components'] = nx.number_connected_components(self.graph)
        
        # Degree statistics
        degrees = dict(self.graph.degree())
        if degrees:
            stats['avg_degree'] = np.mean(list(degrees.values()))
            stats['max_degree'] = max(degrees.values())
            stats['min_degree'] = min(degrees.values())
        
        return stats
    
    def export_graph(self, filepath: str, format: str = 'gexf') -> None:
        """
        Export graph to file.
        
        Args:
            filepath: Path to save the graph
            format: File format ('gexf', 'graphml', 'gml', 'edgelist')
        """
        if format == 'gexf':
            nx.write_gexf(self.graph, filepath)
        elif format == 'graphml':
            nx.write_graphml(self.graph, filepath)
        elif format == 'gml':
            nx.write_gml(self.graph, filepath)
        elif format == 'edgelist':
            nx.write_edgelist(self.graph, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_graph(self, filepath: str, format: str = 'gexf') -> nx.Graph:
        """
        Import graph from file.
        
        Args:
            filepath: Path to the graph file
            format: File format
            
        Returns:
            Loaded graph
        """
        if format == 'gexf':
            self.graph = nx.read_gexf(filepath)
        elif format == 'graphml':
            self.graph = nx.read_graphml(filepath)
        elif format == 'gml':
            self.graph = nx.read_gml(filepath)
        elif format == 'edgelist':
            self.graph = nx.read_edgelist(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return self.graph


def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a graph.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary with various graph metrics
    """
    metrics = {
        'basic': {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_directed': graph.is_directed()
        }
    }
    
    # Only compute expensive metrics for reasonably sized graphs
    if graph.number_of_nodes() < 10000:
        try:
            metrics['clustering'] = nx.average_clustering(graph.to_undirected())
        except:
            metrics['clustering'] = None
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from src.data_processing import generate_synthetic_transactions
    
    # Generate synthetic data
    transactions = generate_synthetic_transactions(n_accounts=50, n_transactions=500)
    
    # Build graph
    tg = TransactionGraph(directed=True)
    tg.load_data(transactions)
    graph = tg.build()
    
    # Get statistics
    stats = tg.get_basic_stats()
    print("Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
