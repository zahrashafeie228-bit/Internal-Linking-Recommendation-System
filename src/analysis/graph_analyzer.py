"""
NetworkX graph analysis for internal link structure.
"""

from typing import Dict, List, Tuple
import networkx as nx
from src.models import InternalLinkData
from src.utils.helpers import setup_logger

logger = setup_logger("graph_analyzer")


class GraphAnalyzer:
    """NetworkX directed graph modeling for web internal link structures."""

    def __init__(self, pages: List[Dict[str, str]], existing_links: List[InternalLinkData]):
        self.graph = nx.DiGraph()
        
        # Add nodes
        for p in pages:
            url = p["url"]
            self.graph.add_node(url, title=p.get("title", ""))

        # Add directed edges
        for link in existing_links:
            src = link.source_url
            tgt = link.target_url
            if self.graph.has_node(src) and self.graph.has_node(tgt):
                self.graph.add_edge(src, tgt)

    def compute_metrics(self) -> Dict[str, any]:
        """
        Compute graph metrics: in-degree, out-degree, PageRank, connected components.
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "pagerank": {},
                "in_degree": {},
                "out_degree": {},
                "connected_components_count": 0,
                "density": 0.0,
            }

        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())

        # PageRank computation
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except Exception as e:
            logger.warning(f"PageRank computation error: {e}")
            pagerank = {n: 1.0 / len(self.graph) for n in self.graph.nodes()}

        # Weakly connected components
        try:
            num_components = nx.number_weakly_connected_components(self.graph)
        except Exception:
            num_components = 1

        density = nx.density(self.graph)

        return {
            "pagerank": pagerank,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "connected_components_count": num_components,
            "density": round(density, 4),
        }
