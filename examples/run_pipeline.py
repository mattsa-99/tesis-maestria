"""
Example script demonstrating the fraud detection pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import generate_synthetic_transactions
from src.graph_construction import TransactionGraph
from src.topology_analysis import TopologyAnalyzer
from src.anomaly_detection import AnomalyDetector, detect_fraud_patterns
from src.visualization import create_comprehensive_report


def main():
    """Run the complete fraud detection pipeline."""
    
    print("=" * 70)
    print("FRAUD DETECTION PIPELINE - TOPOLOGY AND DIRECTED GRAPHS")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic transaction data...")
    transactions = generate_synthetic_transactions(
        n_accounts=100,
        n_transactions=1500,
        fraud_ratio=0.05,
        random_state=42
    )
    print(f"  ✓ Generated {len(transactions)} transactions")
    print(f"  ✓ Fraud rate: {transactions['is_fraud'].mean():.2%}")
    
    # Step 2: Build transaction graph
    print("\n[2/5] Building transaction graph...")
    tg = TransactionGraph(directed=True)
    tg.load_data(transactions)
    graph = tg.build()
    stats = tg.get_basic_stats()
    print(f"  ✓ Nodes: {stats['nodes']}")
    print(f"  ✓ Edges: {stats['edges']}")
    print(f"  ✓ Density: {stats['density']:.4f}")
    
    # Step 3: Analyze topology
    print("\n[3/5] Analyzing graph topology...")
    analyzer = TopologyAnalyzer(graph)
    centrality = analyzer.compute_centrality_measures()
    communities = analyzer.detect_communities()
    hubs = analyzer.identify_hubs()
    print(f"  ✓ Computed {len(centrality)} centrality measures")
    print(f"  ✓ Detected {len(set(communities.values()))} communities")
    print(f"  ✓ Identified {len(hubs)} hub nodes")
    
    # Step 4: Detect anomalies
    print("\n[4/5] Detecting anomalies...")
    detector = AnomalyDetector(graph, contamination=0.05)
    anomaly_scores = detector.detect_node_anomalies()
    patterns = detector.detect_anomalous_patterns()
    top_anomalies = detector.get_top_anomalies(10)
    
    print(f"  ✓ Analyzed {len(anomaly_scores)} nodes for anomalies")
    print(f"  ✓ Star patterns: {len(patterns['star_patterns'])}")
    print(f"  ✓ Anomalous cycles: {len(patterns['anomalous_cycles'])}")
    
    print("\n  Top 5 most anomalous nodes:")
    for i, (node, score) in enumerate(top_anomalies[:5], 1):
        print(f"    {i}. {node}: score={score:.4f}")
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    try:
        create_comprehensive_report(
            graph=graph,
            transactions=transactions,
            anomaly_scores=anomaly_scores,
            centrality_measures=centrality,
            save_dir='results/'
        )
        print("  ✓ Visualizations saved to results/")
    except Exception as e:
        print(f"  ⚠ Could not generate all visualizations: {e}")
    
    # Export results
    print("\nExporting results...")
    tg.export_graph('results/transaction_graph.gexf', format='gexf')
    print("  ✓ Graph exported to results/transaction_graph.gexf")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review visualizations in the results/ directory")
    print("  2. Open notebooks/01_exploratory_analysis.ipynb for detailed analysis")
    print("  3. Customize parameters in config/config.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
