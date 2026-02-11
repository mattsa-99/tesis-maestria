# Configuration file for the thesis project

# Data Processing
DATA_CONFIG = {
    'raw_data_path': 'data/raw/',
    'processed_data_path': 'data/processed/',
    'example_data_path': 'data/examples/',
}

# Graph Construction
GRAPH_CONFIG = {
    'directed': True,
    'weight_attribute': 'amount',
    'aggregation': 'sum',  # 'sum', 'mean', or 'count'
}

# Topology Analysis
TOPOLOGY_CONFIG = {
    'compute_expensive_metrics': False,  # Set to True for smaller graphs
    'max_cycle_length': 10,
    'hub_percentile': 90,
}

# Anomaly Detection
ANOMALY_CONFIG = {
    'contamination': 0.05,  # Expected fraud rate
    'method': 'isolation_forest',  # 'isolation_forest' or 'statistical'
    'threshold_percentile': 95,
}

# Visualization
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'default_layout': 'spring',
    'node_size': 300,
    'save_dpi': 300,
}

# Results
RESULTS_CONFIG = {
    'output_dir': 'results/',
    'save_graphs': True,
    'save_visualizations': True,
    'graph_format': 'gexf',  # 'gexf', 'graphml', 'gml', 'edgelist'
}
