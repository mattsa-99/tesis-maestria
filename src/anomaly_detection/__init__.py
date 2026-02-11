"""
Anomaly Detection Module
Detects anomalous patterns in transaction graphs using topology and machine learning.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class AnomalyDetector:
    """
    Detects anomalies in financial transaction graphs.
    """
    
    def __init__(self, graph: nx.Graph, contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            graph: NetworkX graph to analyze
            contamination: Expected proportion of anomalies (for Isolation Forest)
        """
        self.graph = graph
        self.contamination = contamination
        self.feature_matrix: Optional[np.ndarray] = None
        self.node_features: Optional[pd.DataFrame] = None
        self.anomaly_scores: Optional[Dict[str, float]] = None
        self.model = None
        
    def extract_node_features(self) -> pd.DataFrame:
        """
        Extract topological features for each node.
        
        Returns:
            DataFrame with node features
        """
        features = {}
        nodes = list(self.graph.nodes())
        
        # Degree features
        degrees = dict(self.graph.degree())
        
        if self.graph.is_directed():
            in_degrees = dict(self.graph.in_degree())
            out_degrees = dict(self.graph.out_degree())
        
        # Clustering coefficient
        clustering = nx.clustering(self.graph.to_undirected())
        
        # PageRank (for directed graphs)
        if self.graph.is_directed():
            try:
                pagerank = nx.pagerank(self.graph)
            except:
                pagerank = {node: 0 for node in nodes}
        
        # Build feature dictionary
        for node in nodes:
            node_features = {
                'degree': degrees.get(node, 0),
                'clustering': clustering.get(node, 0)
            }
            
            if self.graph.is_directed():
                node_features['in_degree'] = in_degrees.get(node, 0)
                node_features['out_degree'] = out_degrees.get(node, 0)
                node_features['degree_ratio'] = (
                    out_degrees.get(node, 0) / max(in_degrees.get(node, 1), 1)
                )
                node_features['pagerank'] = pagerank.get(node, 0)
            
            # Transaction volume (sum of edge weights)
            if self.graph.is_directed():
                out_volume = sum(
                    self.graph[node][neighbor].get('weight', 0)
                    for neighbor in self.graph.successors(node)
                )
                in_volume = sum(
                    self.graph[neighbor][node].get('weight', 0)
                    for neighbor in self.graph.predecessors(node)
                )
                node_features['out_volume'] = out_volume
                node_features['in_volume'] = in_volume
                node_features['volume_ratio'] = out_volume / max(in_volume, 1)
            else:
                volume = sum(
                    self.graph[node][neighbor].get('weight', 0)
                    for neighbor in self.graph.neighbors(node)
                )
                node_features['volume'] = volume
            
            features[node] = node_features
        
        self.node_features = pd.DataFrame.from_dict(features, orient='index')
        return self.node_features
    
    def extract_edge_features(self) -> pd.DataFrame:
        """
        Extract features for each edge (transaction).
        
        Returns:
            DataFrame with edge features
        """
        edge_features = []
        
        for source, target, data in self.graph.edges(data=True):
            features = {
                'source': source,
                'target': target,
                'weight': data.get('weight', 0),
                'transaction_count': data.get('transaction_count', 1)
            }
            
            # Source node features
            features['source_degree'] = self.graph.degree(source)
            if self.graph.is_directed():
                features['source_out_degree'] = self.graph.out_degree(source)
                features['target_in_degree'] = self.graph.in_degree(target)
            
            # Edge reciprocity (exists in both directions?)
            if self.graph.is_directed():
                features['is_reciprocal'] = self.graph.has_edge(target, source)
            
            edge_features.append(features)
        
        return pd.DataFrame(edge_features)
    
    def detect_node_anomalies(self, method: str = 'isolation_forest') -> Dict[str, float]:
        """
        Detect anomalous nodes.
        
        Args:
            method: Detection method ('isolation_forest', 'statistical')
            
        Returns:
            Dictionary mapping nodes to anomaly scores
        """
        if self.node_features is None:
            self.extract_node_features()
        
        if method == 'isolation_forest':
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.node_features)
            
            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            predictions = self.model.fit_predict(features_scaled)
            
            # Get anomaly scores (lower is more anomalous)
            scores = self.model.score_samples(features_scaled)
            
            # Convert to dictionary
            self.anomaly_scores = {
                node: float(score)
                for node, score in zip(self.node_features.index, scores)
            }
            
        elif method == 'statistical':
            # Use statistical approach (z-scores)
            self.anomaly_scores = {}
            
            for node in self.node_features.index:
                node_score = 0
                for col in self.node_features.columns:
                    values = self.node_features[col]
                    mean = values.mean()
                    std = values.std()
                    
                    if std > 0:
                        z_score = abs((self.node_features.loc[node, col] - mean) / std)
                        node_score += z_score
                
                self.anomaly_scores[node] = -node_score  # Negative for consistency
        
        return self.anomaly_scores
    
    def detect_edge_anomalies(self, threshold_percentile: float = 95) -> List[Tuple[str, str]]:
        """
        Detect anomalous edges (transactions).
        
        Args:
            threshold_percentile: Percentile threshold for anomaly
            
        Returns:
            List of anomalous edges (source, target)
        """
        edge_features = self.extract_edge_features()
        
        # Anomalous based on weight
        weight_threshold = np.percentile(edge_features['weight'], threshold_percentile)
        anomalous_edges = edge_features[edge_features['weight'] > weight_threshold]
        
        return list(zip(anomalous_edges['source'], anomalous_edges['target']))
    
    def detect_anomalous_patterns(self) -> Dict[str, List]:
        """
        Detect various anomalous patterns in the graph.
        
        Returns:
            Dictionary with different types of anomalies
        """
        patterns = {
            'isolated_nodes': [],
            'star_patterns': [],
            'anomalous_cycles': [],
            'sudden_spikes': []
        }
        
        # Isolated nodes (potential dormant accounts suddenly active)
        for node in self.graph.nodes():
            if self.graph.degree(node) == 0:
                patterns['isolated_nodes'].append(node)
        
        # Star patterns (one node with many connections - money mule)
        degree_threshold = np.percentile(list(dict(self.graph.degree()).values()), 95)
        for node in self.graph.nodes():
            if self.graph.degree(node) > degree_threshold:
                patterns['star_patterns'].append(node)
        
        # Cycles (circular transactions)
        if self.graph.is_directed():
            try:
                cycles = list(nx.simple_cycles(self.graph))
                # Focus on small cycles (likely fraudulent)
                patterns['anomalous_cycles'] = [c for c in cycles if len(c) <= 5]
            except:
                patterns['anomalous_cycles'] = []
        
        return patterns
    
    def get_top_anomalies(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N anomalous nodes.
        
        Args:
            n: Number of top anomalies to return
            
        Returns:
            List of (node_id, score) tuples
        """
        if self.anomaly_scores is None:
            self.detect_node_anomalies()
        
        sorted_anomalies = sorted(
            self.anomaly_scores.items(),
            key=lambda x: x[1]
        )
        
        return sorted_anomalies[:n]
    
    def explain_anomaly(self, node: str) -> Dict[str, Any]:
        """
        Explain why a node is considered anomalous.
        
        Args:
            node: Node ID
            
        Returns:
            Dictionary with explanation
        """
        if self.node_features is None:
            self.extract_node_features()
        
        if node not in self.node_features.index:
            raise ValueError(f"Node {node} not found")
        
        node_data = self.node_features.loc[node]
        
        explanation = {
            'node': node,
            'features': node_data.to_dict(),
            'anomaly_score': self.anomaly_scores.get(node) if self.anomaly_scores else None,
            'comparisons': {}
        }
        
        # Compare to averages
        for col in self.node_features.columns:
            mean = self.node_features[col].mean()
            std = self.node_features[col].std()
            value = node_data[col]
            
            if std > 0:
                z_score = (value - mean) / std
                explanation['comparisons'][col] = {
                    'value': float(value),
                    'mean': float(mean),
                    'std': float(std),
                    'z_score': float(z_score),
                    'is_outlier': abs(z_score) > 2
                }
        
        return explanation


def detect_fraud_patterns(graph: nx.Graph, 
                         transactions: pd.DataFrame,
                         contamination: float = 0.05) -> Dict[str, Any]:
    """
    Comprehensive fraud detection using multiple methods.
    
    Args:
        graph: Transaction graph
        transactions: Original transaction data
        contamination: Expected fraud rate
        
    Returns:
        Dictionary with detection results
    """
    detector = AnomalyDetector(graph, contamination=contamination)
    
    results = {
        'node_anomalies': detector.detect_node_anomalies(),
        'edge_anomalies': detector.detect_edge_anomalies(),
        'patterns': detector.detect_anomalous_patterns(),
        'top_10_anomalies': detector.get_top_anomalies(10)
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.graph_construction import TransactionGraph
    from src.data_processing import generate_synthetic_transactions
    
    # Generate synthetic data with fraud
    transactions = generate_synthetic_transactions(
        n_accounts=100,
        n_transactions=1000,
        fraud_ratio=0.05
    )
    
    # Build graph
    tg = TransactionGraph(directed=True)
    tg.load_data(transactions)
    graph = tg.build()
    
    # Detect anomalies
    detector = AnomalyDetector(graph, contamination=0.05)
    anomalies = detector.detect_node_anomalies()
    
    # Get top anomalies
    top_anomalies = detector.get_top_anomalies(5)
    print("\nTop 5 anomalous nodes:")
    for node, score in top_anomalies:
        print(f"  {node}: {score:.4f}")
        
    # Detect patterns
    patterns = detector.detect_anomalous_patterns()
    print(f"\nAnomalous patterns found:")
    print(f"  Star patterns: {len(patterns['star_patterns'])}")
    print(f"  Anomalous cycles: {len(patterns['anomalous_cycles'])}")
