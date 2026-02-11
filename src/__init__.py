"""
Financial Fraud Detection using Topology and Directed Graphs
Master's Thesis Project
"""

__version__ = "0.1.0"
__author__ = "Matías Sánchez"
__title__ = "Detección de Fraude Financiero mediante Topología y Grafos Dirigidos"

from src.graph_construction import TransactionGraph
from src.anomaly_detection import AnomalyDetector

__all__ = [
    "TransactionGraph",
    "AnomalyDetector",
]
