//!
//! Common definition and utility functions for graphs
//!

use fnv::FnvHashSet as HashSet;
use petgraph::prelude::*;

///
/// FloatWeight is generalized type of f64.
///
/// It represents a type for which
/// * f64 convertable
/// * epsilon is defined
///
pub trait FloatWeight {
    fn float_weight(&self) -> f64;
    fn epsilon() -> f64;
}

impl FloatWeight for f64 {
    fn float_weight(&self) -> f64 {
        *self
    }
    fn epsilon() -> f64 {
        f64::EPSILON
    }
}

///
/// Calculate total weight of path (a list of edges)
///
pub fn total_weight<N, E: FloatWeight>(graph: &DiGraph<N, E>, edges: &[EdgeIndex]) -> f64 {
    edges
        .iter()
        .map(|&e| {
            let ew = graph.edge_weight(e).unwrap();
            ew.float_weight()
        })
        .sum()
}

///
/// short-hand of `NodeIndex::new`
///
pub fn ni(index: usize) -> NodeIndex {
    NodeIndex::new(index)
}

///
/// short-hand of `EdgeIndex::new`
///
pub fn ei(index: usize) -> EdgeIndex {
    EdgeIndex::new(index)
}

///
/// Determine if a cycle given by edges is a negative cycle or not.
///
pub fn is_negative_cycle<N, E: FloatWeight>(graph: &DiGraph<N, E>, edges: &[EdgeIndex]) -> bool {
    total_weight(graph, edges) < 0.0
}

///
/// Check if the edge list forms a cycle in the graph.
///
/// * For all two adjacent edges e1 and e2, target(e1) and source(e2) is the same node.
///
pub fn is_cycle<N, E>(graph: &DiGraph<N, E>, edges: &[EdgeIndex]) -> bool {
    let n = edges.len();
    (0..n).all(|i| {
        let e1 = edges[i];
        let e2 = edges[(i + 1) % n];

        //   e1           e2
        // -----> v1/v2 ------>
        let (_, v1) = graph
            .edge_endpoints(e1)
            .expect("the edge is not in the graph");
        let (v2, _) = graph
            .edge_endpoints(e2)
            .expect("the edge is not in the graph");

        v1 == v2
    })
}

///
/// determine if the path (= a list of nodes) is node-simple
///
pub fn is_node_simple<N, E>(graph: &DiGraph<N, E>, nodes: &[NodeIndex]) -> bool {
    let mut used: HashSet<NodeIndex> = HashSet::default();
    for &node in nodes {
        if used.contains(&node) {
            return false;
        } else {
            used.insert(node);
        }
    }
    return true;
}

///
/// determine if the path (= a list of edges) is edge-simple
///
pub fn is_edge_simple<N, E>(graph: &DiGraph<N, E>, edges: &[EdgeIndex]) -> bool {
    let mut used: HashSet<EdgeIndex> = HashSet::default();
    for &edge in edges {
        if used.contains(&edge) {
            return false;
        } else {
            used.insert(edge);
        }
    }
    return true;
}

/// Find the minimum weight edge among all parallel edges between v and w
/// Input: two nodes (v,w) in a graph
/// Output: minimum weight edge among all parallel edge (v,w)
///
/// (Used in `node_list_to_edge_list`)
fn pick_minimum_weight_edge<N, E: FloatWeight>(
    graph: &DiGraph<N, E>,
    v: NodeIndex,
    w: NodeIndex,
) -> EdgeIndex {
    let er = graph
        .edges_connecting(v, w)
        .min_by(|e1, e2| {
            let w1 = e1.weight().float_weight();
            let w2 = e2.weight().float_weight();
            w1.partial_cmp(&w2).unwrap()
        })
        .unwrap_or_else(|| {
            panic!(
                "there is no edge between node {} and {}",
                v.index(),
                w.index()
            )
        });
    let e = er.id();
    e
}

///
/// Convert "a cycle as nodes [NodeIndex]" into "a cycle as edges [EdgeIndex]",
/// by choosing the minimum weight edge if there are parallel edges
///
pub fn node_list_to_edge_list<N, E: FloatWeight>(
    graph: &DiGraph<N, E>,
    nodes: &[NodeIndex],
) -> Vec<EdgeIndex> {
    nodes_to_edges(graph, nodes, |graph, v, w| {
        pick_minimum_weight_edge(graph, v, w)
    })
}

///
/// convert a list of nodes into a list of edges
/// by using `edge_picker` (that chooses an edge from (v, w))
///
pub fn nodes_to_edges<N, E, F>(
    graph: &DiGraph<N, E>,
    nodes: &[NodeIndex],
    edge_picker: F,
) -> Vec<EdgeIndex>
where
    F: Fn(&DiGraph<N, E>, NodeIndex, NodeIndex) -> EdgeIndex,
{
    let mut edges = Vec::new();
    let n = nodes.len();

    // convert (nodes[i], nodes[i+1]) into an edge
    for i in 0..n {
        let v = nodes[i];
        let w = nodes[(i + 1) % n];
        let edge = edge_picker(graph, v, w);
        edges.push(edge);
    }

    edges
}

///
/// Convert edge list into node list
///
pub fn edge_list_to_node_list<N, E>(graph: &DiGraph<N, E>, edges: &[EdgeIndex]) -> Vec<NodeIndex> {
    let mut nodes = Vec::new();

    // (1) add first edge
    let (v0, _) = graph
        .edge_endpoints(edges[0])
        .expect("edge is not in the graph");
    nodes.push(v0);

    // (2) add target node of each edge
    for &edge in edges.iter() {
        let (_, v) = graph
            .edge_endpoints(edge)
            .expect("edge is not in the graph");
        nodes.push(v);
    }

    nodes
}

///
/// Convert cycle of edge into node list
///
pub fn edge_cycle_to_node_cycle<N, E>(
    graph: &DiGraph<N, E>,
    cycle: &[EdgeIndex],
) -> Vec<NodeIndex> {
    cycle
        .iter()
        .map(|&edge| {
            let (v, _) = graph
                .edge_endpoints(edge)
                .expect("edge is not in the graph");
            v
        })
        .collect()
}
