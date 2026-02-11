"""
Visualization Module
Visualizes transaction graphs and anomaly detection results.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class GraphVisualizer:
    """
    Visualizes transaction graphs and analysis results.
    """
    
    def __init__(self, graph: nx.Graph, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the graph visualizer.
        
        Args:
            graph: NetworkX graph to visualize
            figsize: Figure size for plots
        """
        self.graph = graph
        self.figsize = figsize
        
    def plot_graph(self, 
                   node_colors: Optional[Dict[str, Any]] = None,
                   edge_colors: Optional[Dict[Tuple, Any]] = None,
                   node_size: int = 300,
                   layout: str = 'spring',
                   title: str = 'Transaction Graph',
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the graph with customizable styling.
        
        Args:
            node_colors: Dictionary mapping nodes to colors or values
            edge_colors: Dictionary mapping edges to colors or values
            node_size: Size of nodes
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Prepare node colors
        if node_colors:
            node_color_list = [node_colors.get(node, 0.5) for node in self.graph.nodes()]
        else:
            node_color_list = 'lightblue'
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_color_list,
            node_size=node_size,
            ax=ax,
            cmap=plt.cm.RdYlGn_r if node_colors else None
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            alpha=0.5,
            arrows=self.graph.is_directed(),
            arrowsize=10,
            ax=ax
        )
        
        # Draw labels for smaller graphs
        if self.graph.number_of_nodes() < 50:
            nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_anomaly_scores(self,
                           anomaly_scores: Dict[str, float],
                           top_n: int = 20,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot anomaly scores for nodes.
        
        Args:
            anomaly_scores: Dictionary mapping nodes to anomaly scores
            top_n: Number of top anomalies to display
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Sort by anomaly score
        sorted_scores = sorted(anomaly_scores.items(), key=lambda x: x[1])[:top_n]
        nodes, scores = zip(*sorted_scores)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['red' if s < np.percentile(list(anomaly_scores.values()), 10) else 'orange' 
                 for s in scores]
        
        ax.barh(range(len(nodes)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes)
        ax.set_xlabel('Anomaly Score (lower = more anomalous)', fontsize=12)
        ax.set_title(f'Top {top_n} Anomalous Nodes', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_degree_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot degree distribution of the graph.
        
        Args:
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        degrees = [d for _, d in self.graph.degree()]
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        axes[0].hist(degrees, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Degree', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Degree Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Log-log plot
        degree_count = pd.Series(degrees).value_counts().sort_index()
        axes[1].loglog(degree_count.index, degree_count.values, 'o', color='coral', alpha=0.7)
        axes[1].set_xlabel('Degree (log scale)', fontsize=12)
        axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
        axes[1].set_title('Degree Distribution (Log-Log)', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_centrality_comparison(self,
                                  centrality_measures: Dict[str, Dict[str, float]],
                                  top_n: int = 10,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different centrality measures.
        
        Args:
            centrality_measures: Dictionary of centrality measure dictionaries
            top_n: Number of top nodes to display
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        n_measures = len(centrality_measures)
        fig, axes = plt.subplots(1, n_measures, figsize=(6 * n_measures, 6))
        
        if n_measures == 1:
            axes = [axes]
        
        for idx, (measure_name, measure_values) in enumerate(centrality_measures.items()):
            if measure_values is None:
                continue
                
            # Get top N nodes
            sorted_nodes = sorted(measure_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
            nodes, values = zip(*sorted_nodes)
            
            axes[idx].barh(range(len(nodes)), values, color='steelblue', alpha=0.7)
            axes[idx].set_yticks(range(len(nodes)))
            axes[idx].set_yticklabels(nodes, fontsize=8)
            axes[idx].set_xlabel('Centrality Score', fontsize=10)
            axes[idx].set_title(f'{measure_name.replace("_", " ").title()}', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_community_structure(self,
                                communities: Dict[str, int],
                                layout: str = 'spring',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize community structure.
        
        Args:
            communities: Dictionary mapping nodes to community IDs
            layout: Layout algorithm
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Create color map
        unique_communities = set(communities.values())
        color_map = plt.cm.get_cmap('tab20', len(unique_communities))
        node_colors = [color_map(communities.get(node, 0)) for node in self.graph.nodes()]
        
        # Draw graph
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=300,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            alpha=0.3,
            arrows=self.graph.is_directed(),
            ax=ax
        )
        
        ax.set_title(f'Community Structure ({len(unique_communities)} communities)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_transaction_timeline(self,
                                 transactions: pd.DataFrame,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot transaction timeline.
        
        Args:
            transactions: DataFrame with transaction data including timestamp
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if 'timestamp' not in transactions.columns:
            raise ValueError("Transactions must have 'timestamp' column")
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # Transactions over time
        transactions['date'] = pd.to_datetime(transactions['timestamp']).dt.date
        daily_counts = transactions.groupby('date').size()
        
        axes[0].plot(daily_counts.index, daily_counts.values, color='steelblue', linewidth=2)
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Number of Transactions', fontsize=12)
        axes[0].set_title('Transaction Volume Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Transaction amounts over time
        if 'amount' in transactions.columns:
            daily_amounts = transactions.groupby('date')['amount'].sum()
            axes[1].plot(daily_amounts.index, daily_amounts.values, color='coral', linewidth=2)
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].set_ylabel('Total Amount', fontsize=12)
            axes[1].set_title('Transaction Amount Over Time', fontsize=14, fontweight='bold')
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_comprehensive_report(graph: nx.Graph,
                                transactions: pd.DataFrame,
                                anomaly_scores: Dict[str, float],
                                centrality_measures: Dict[str, Dict[str, float]],
                                save_dir: str = 'results/') -> None:
    """
    Create a comprehensive visualization report.
    
    Args:
        graph: Transaction graph
        transactions: Transaction data
        anomaly_scores: Anomaly scores for nodes
        centrality_measures: Various centrality measures
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = GraphVisualizer(graph)
    
    # 1. Graph overview
    visualizer.plot_graph(
        node_colors=anomaly_scores,
        title='Transaction Graph with Anomaly Scores',
        save_path=f'{save_dir}/graph_overview.png'
    )
    plt.close()
    
    # 2. Anomaly scores
    visualizer.plot_anomaly_scores(
        anomaly_scores,
        top_n=20,
        save_path=f'{save_dir}/anomaly_scores.png'
    )
    plt.close()
    
    # 3. Degree distribution
    visualizer.plot_degree_distribution(
        save_path=f'{save_dir}/degree_distribution.png'
    )
    plt.close()
    
    # 4. Centrality comparison
    visualizer.plot_centrality_comparison(
        centrality_measures,
        save_path=f'{save_dir}/centrality_comparison.png'
    )
    plt.close()
    
    print(f"Visualizations saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    from src.graph_construction import TransactionGraph
    from src.data_processing import generate_synthetic_transactions
    from src.anomaly_detection import AnomalyDetector
    
    # Generate and build graph
    transactions = generate_synthetic_transactions(n_accounts=50, n_transactions=500)
    tg = TransactionGraph(directed=True)
    tg.load_data(transactions)
    graph = tg.build()
    
    # Detect anomalies
    detector = AnomalyDetector(graph)
    anomaly_scores = detector.detect_node_anomalies()
    
    # Visualize
    visualizer = GraphVisualizer(graph)
    fig = visualizer.plot_graph(node_colors=anomaly_scores, title='Transaction Graph')
    plt.show()
