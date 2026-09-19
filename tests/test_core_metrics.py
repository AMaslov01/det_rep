from det_rep.core.extract import Graph
from det_rep.core.matching import DictEmbedder, RefGraph
from det_rep.core.metrics import score_response
from det_rep.core.config import Config


def test_ported_historical_strict_formula_and_empty_graph_policy():
    settings = Config({
        "entity_sim_threshold": 0.9, "relation_sim_threshold": 0.75,
        "allow_substring_match": True, "direction_sensitive_edges": True,
        "inverse_edge_match": False, "min_substring_chars": 2, "stopwords": [],
    })
    source = Graph({"Alice", "Paris"}, {("Alice", "lives in", "Paris")})
    ref = RefGraph(source.entities, source.relations, settings, DictEmbedder(dim=16))
    exact = score_response(source, ref, source, Graph.empty())
    assert exact.EG == exact.RP_strict == 1.0
    assert exact.h_for_mode(0.5, "strict") == 0.0
    wrong = Graph({"Alice", "London"}, {("Alice", "lives in", "London")})
    result = score_response(wrong, ref, source, Graph.empty())
    assert result.EG == 0.5 and result.RP_strict == 0.0
    assert result.h_for_mode(0.5, "strict") == 0.75
    empty = score_response(Graph.empty(), ref, source, Graph.empty())
    assert empty.unscorable and empty.h_for_mode(0.5, "strict") is None
