"""
Topology Analysis Module
Analyzes topological properties of transaction graphs.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class TopologyAnalyzer:
    """
    Analyzes topological properties of transaction graphs.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize the topology analyzer.
        
        Args:
            graph: NetworkX graph to analyze
        """
        self.graph = graph
        self.metrics: Dict[str, Any] = {}
        
    def compute_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Compute various centrality measures for all nodes.
        
        Returns:
            Dictionary of centrality measures
        """
        centrality = {}
        
        # Degree centrality
        centrality['degree'] = nx.degree_centrality(self.graph)
        
        if self.graph.is_directed():
            centrality['in_degree'] = nx.in_degree_centrality(self.graph)
            centrality['out_degree'] = nx.out_degree_centrality(self.graph)
        
        # Only compute expensive metrics for smaller graphs
        if self.graph.number_of_nodes() < 5000:
            try:
                # Betweenness centrality (important for detecting intermediaries)
                centrality['betweenness'] = nx.betweenness_centrality(self.graph)
                
                # Closeness centrality
                centrality['closeness'] = nx.closeness_centrality(self.graph)
                
                # PageRank (for directed graphs)
                if self.graph.is_directed():
                    centrality['pagerank'] = nx.pagerank(self.graph)
                
                # Eigenvector centrality
                try:
                    centrality['eigenvector'] = nx.eigenvector_centrality(self.graph, max_iter=100)
                except:
                    centrality['eigenvector'] = None
            except Exception as e:
                print(f"Warning: Could not compute some centrality measures: {e}")
        
        self.metrics['centrality'] = centrality
        return centrality
    
    def detect_communities(self, method: str = 'louvain') -> Dict[str, int]:
        """
        Detect communities in the graph.
        
        Args:
            method: Community detection method ('louvain', 'label_propagation', 'greedy')
            
        Returns:
            Dictionary mapping nodes to community IDs
        """
        if method == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(self.graph.to_undirected())
            except ImportError:
                print("Warning: python-louvain not installed, using label propagation")
                method = 'label_propagation'
        
        if method == 'label_propagation':
            communities_gen = nx.community.label_propagation_communities(self.graph.to_undirected())
            communities = {}
            for i, community in enumerate(communities_gen):
                for node in community:
                    communities[node] = i
        elif method == 'greedy':
            communities_gen = nx.community.greedy_modularity_communities(self.graph.to_undirected())
            communities = {}
            for i, community in enumerate(communities_gen):
                for node in community:
                    communities[node] = i
        
        self.metrics['communities'] = communities
        return communities
    
    def find_strongly_connected_components(self) -> List[List[str]]:
        """
        Find strongly connected components (for directed graphs).
        
        Returns:
            List of strongly connected components
        """
        if not self.graph.is_directed():
            raise ValueError("Graph must be directed for strongly connected components")
        
        sccs = list(nx.strongly_connected_components(self.graph))
        self.metrics['sccs'] = sccs
        return sccs
    
    def detect_cycles(self, max_length: Optional[int] = 10) -> List[List[str]]:
        """
        Detect cycles in the graph (important for fraud detection).
        
        Args:
            max_length: Maximum cycle length to search for
            
        Returns:
            List of cycles
        """
        cycles = []
        
        if self.graph.is_directed():
            try:
                # Find simple cycles
                all_cycles = list(nx.simple_cycles(self.graph))
                
                # Filter by length if specified
                if max_length:
                    cycles = [c for c in all_cycles if len(c) <= max_length]
                else:
                    cycles = all_cycles
            except:
                cycles = []
        else:
            # For undirected graphs, use cycle basis
            cycles = nx.cycle_basis(self.graph)
        
        self.metrics['cycles'] = cycles
        return cycles
    
    def compute_flow_metrics(self) -> Dict[str, Any]:
        """
        Compute flow-related metrics for the graph.
        
        Returns:
            Dictionary with flow metrics
        """
        flow_metrics = {}
        
        # In/out degree for directed graphs
        if self.graph.is_directed():
            in_degrees = dict(self.graph.in_degree())
            out_degrees = dict(self.graph.out_degree())
            
            # Compute flow imbalance (difference between in and out)
            flow_imbalance = {
                node: out_degrees.get(node, 0) - in_degrees.get(node, 0)
                for node in self.graph.nodes()
            }
            
            flow_metrics['in_degree'] = in_degrees
            flow_metrics['out_degree'] = out_degrees
            flow_metrics['flow_imbalance'] = flow_imbalance
            
            # Identify sources (high out-degree, low in-degree)
            sources = [node for node, imb in flow_imbalance.items() if imb > np.percentile(list(flow_imbalance.values()), 90)]
            flow_metrics['sources'] = sources
            
            # Identify sinks (high in-degree, low out-degree)
            sinks = [node for node, imb in flow_imbalance.items() if imb < np.percentile(list(flow_imbalance.values()), 10)]
            flow_metrics['sinks'] = sinks
        
        self.metrics['flow'] = flow_metrics
        return flow_metrics
    
    def compute_clustering_coefficient(self) -> Dict[str, float]:
        """
        Compute clustering coefficient for each node.
        
        Returns:
            Dictionary mapping nodes to clustering coefficients
        """
        # Convert to undirected for clustering
        undirected = self.graph.to_undirected()
        clustering = nx.clustering(undirected)
        
        self.metrics['clustering'] = {
            'node_clustering': clustering,
            'average': nx.average_clustering(undirected)
        }
        
        return clustering
    
    def identify_hubs(self, threshold_percentile: float = 90) -> List[str]:
        """
        Identify hub nodes based on degree.
        
        Args:
            threshold_percentile: Percentile threshold for hub identification
            
        Returns:
            List of hub node IDs
        """
        degrees = dict(self.graph.degree())
        threshold = np.percentile(list(degrees.values()), threshold_percentile)
        hubs = [node for node, degree in degrees.items() if degree >= threshold]
        
        self.metrics['hubs'] = hubs
        return hubs
    
    def compute_shortest_path_metrics(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Compute shortest path related metrics.
        
        Args:
            sample_size: Sample size for large graphs (None = all pairs)
            
        Returns:
            Dictionary with path metrics
        """
        path_metrics = {}
        
        try:
            if sample_size and self.graph.number_of_nodes() > sample_size:
                # Sample nodes for large graphs
                nodes = list(self.graph.nodes())
                sampled_nodes = np.random.choice(nodes, size=min(sample_size, len(nodes)), replace=False)
                
                path_lengths = []
                for source in sampled_nodes:
                    lengths = nx.single_source_shortest_path_length(self.graph, source)
                    path_lengths.extend(lengths.values())
                
                path_metrics['avg_shortest_path'] = np.mean(path_lengths) if path_lengths else 0
                path_metrics['sampled'] = True
            else:
                # Compute for all connected components
                if self.graph.is_directed():
                    if nx.is_strongly_connected(self.graph):
                        path_metrics['avg_shortest_path'] = nx.average_shortest_path_length(self.graph)
                    else:
                        path_metrics['avg_shortest_path'] = None
                else:
                    if nx.is_connected(self.graph):
                        path_metrics['avg_shortest_path'] = nx.average_shortest_path_length(self.graph)
                    else:
                        path_metrics['avg_shortest_path'] = None
                
                path_metrics['sampled'] = False
            
            # Diameter (if connected)
            try:
                path_metrics['diameter'] = nx.diameter(self.graph)
            except:
                path_metrics['diameter'] = None
                
        except Exception as e:
            print(f"Warning: Could not compute shortest path metrics: {e}")
            path_metrics['error'] = str(e)
        
        self.metrics['paths'] = path_metrics
        return path_metrics
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of all computed metrics.
        
        Returns:
            DataFrame with metric summary
        """
        summary_data = []
        
        for metric_type, values in self.metrics.items():
            if isinstance(values, dict) and metric_type == 'centrality':
                for centrality_type, centrality_values in values.items():
                    if centrality_values:
                        summary_data.append({
                            'metric_type': 'centrality',
                            'metric_name': centrality_type,
                            'mean': np.mean(list(centrality_values.values())),
                            'std': np.std(list(centrality_values.values())),
                            'min': min(centrality_values.values()),
                            'max': max(centrality_values.values())
                        })
        
        return pd.DataFrame(summary_data)


def analyze_graph_topology(graph: nx.Graph, 
                          compute_all: bool = False) -> Dict[str, Any]:
    """
    Perform comprehensive topology analysis.
    
    Args:
        graph: NetworkX graph
        compute_all: Whether to compute all metrics (may be slow for large graphs)
        
    Returns:
        Dictionary with all topology metrics
    """
    analyzer = TopologyAnalyzer(graph)
    
    results = {
        'basic_stats': {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_directed': graph.is_directed()
        }
    }
    
    # Always compute centrality
    results['centrality'] = analyzer.compute_centrality_measures()
    
    if compute_all or graph.number_of_nodes() < 1000:
        try:
            results['communities'] = analyzer.detect_communities()
        except Exception as e:
            print(f"Could not detect communities: {e}")
        
        try:
            results['hubs'] = analyzer.identify_hubs()
        except Exception as e:
            print(f"Could not identify hubs: {e}")
        
        if graph.is_directed():
            try:
                results['cycles'] = analyzer.detect_cycles()
            except Exception as e:
                print(f"Could not detect cycles: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from src.graph_construction import TransactionGraph
    from src.data_processing import generate_synthetic_transactions
    
    # Generate and build graph
    transactions = generate_synthetic_transactions(n_accounts=30, n_transactions=200)
    tg = TransactionGraph(directed=True)
    tg.load_data(transactions)
    graph = tg.build()
    
    # Analyze topology
    analyzer = TopologyAnalyzer(graph)
    centrality = analyzer.compute_centrality_measures()
    
    print("Top 5 nodes by PageRank:")
    if 'pagerank' in centrality and centrality['pagerank']:
        sorted_nodes = sorted(centrality['pagerank'].items(), key=lambda x: x[1], reverse=True)[:5]
        for node, score in sorted_nodes:
            print(f"  {node}: {score:.4f}")
