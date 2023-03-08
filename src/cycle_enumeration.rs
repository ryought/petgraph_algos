//!
//! Cycle enumeration using [Johnson1975]() algorithm
//!
//! both for directed and undirected graphs
//!
//! * Tarjan1972
//! Enumeration of the Elementary Circuits of a Directed Graph
//! https://ecommons.cornell.edu/handle/1813/5941
//!
//! * Johnson1975
//! Finding all the elementary circuits of a directed graph
//!
use super::common::nodes_to_edges;
use super::cycle::Cycle;
use fnv::FnvHashSet as HashSet;
use petgraph::graph::{DiGraph, EdgeIndex, Graph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::VecDeque;

///
/// Enumerate all simple cycles (elementary circuits; a node-simple path starts/ends from the same node)
///
/// Assumptions
/// * graph is strongly-connected.
///     there are path of both directions (v->w and w->v) between any two nodes v and w).
///
/// TODO
/// * If graph has self-loops, output them as simple cycles.
/// * To find simple_cycles in residue graph prohibit (v+ -> v-)-type transitions
///
pub fn simple_cycles_as_nodes<N, E>(graph: &DiGraph<N, E>) -> Vec<Vec<NodeIndex>> {
    // let mut cycles = vec![];
    let n = graph.node_count();
    let ix = |node: NodeIndex| node.index();
    let mut ret: Vec<Vec<NodeIndex>> = Vec::new();
    for start_node in graph.node_indices() {
        // search for circuit starting from node
        //
        let mut blocked = vec![false; n];
        blocked[ix(start_node)] = true;
        // B-list
        let mut b: Vec<HashSet<NodeIndex>> = vec![HashSet::default(); n];
        // abbr of ix() node.index()
        // recursively unblocks nodes connected by B-list
        let unblock =
            |node: NodeIndex, blocked: &mut Vec<bool>, b: &mut Vec<HashSet<NodeIndex>>| {
                let mut nodes = vec![node];
                while let Some(node) = nodes.pop() {
                    blocked[ix(node)] = false;
                    for &bnode in b[ix(node)].iter() {
                        nodes.push(bnode);
                    }
                    b[ix(node)].clear();
                }
            };
        let neighbors = |node: NodeIndex| {
            let ret: Vec<_> = graph
                .neighbors_directed(node, Direction::Outgoing)
                .filter(|next_node| next_node.index() >= start_node.index())
                .collect();
            ret
        };
        let mut path: Vec<NodeIndex> = vec![start_node];
        let mut stack: Vec<(NodeIndex, Vec<NodeIndex>)> = vec![(start_node, neighbors(start_node))];
        let mut closed: HashSet<NodeIndex> = HashSet::default();

        // CIRCUIT(node) routine
        while let Some((node, next_nodes)) = stack.last_mut() {
            // for w in A_k(v)
            if !next_nodes.is_empty() {
                let next_node = next_nodes.pop().unwrap();
                // there are neighbor nodes
                // (1) back to the start node
                if next_node == start_node {
                    // cycle found!
                    ret.push(path.clone());
                    for &node_in_path in path.iter() {
                        closed.insert(node_in_path);
                    }
                }
                // (2) visit a unblocked neighboring node
                if !blocked[ix(next_node)] {
                    path.push(next_node);
                    blocked[ix(next_node)] = true;
                    stack.push((next_node, neighbors(next_node)));
                    closed.remove(&next_node);
                    continue;
                }
            }
            // if f then unblock else put v in b(w)
            if next_nodes.is_empty() {
                // no neighbors
                if closed.contains(node) {
                    // f=true
                    unblock(*node, &mut blocked, &mut b);
                } else {
                    // f=false
                    for neighbor in neighbors(*node) {
                        b[ix(neighbor)].insert(*node);
                    }
                }
                stack.pop();
                path.pop();
            }
        }
    }
    ret
}

///
/// assuming there are no parallel edges
///
pub fn simple_cycles<N, E>(graph: &DiGraph<N, E>) -> Vec<Cycle> {
    simple_cycles_as_nodes(graph)
        .into_iter()
        .map(|nodes| {
            let edges = nodes_to_edges(graph, &nodes, |graph, v, w| {
                // assert!()
                graph.edges_connecting(v, w).next().unwrap().id()
            });
            Cycle::new(edges)
        })
        .collect()
}

///
/// enumerate all cycles whose length is k
///
pub fn simple_k_cycles<N, E>(graph: &DiGraph<N, E>, k: usize) -> Vec<Cycle> {
    simple_k_cycles_with_cond(graph, k, |_, _| true)
}

///
/// enumerate all cycles whose length is k with `is_movable` condition
///
pub fn simple_k_cycles_with_cond<N, E, F>(
    graph: &DiGraph<N, E>,
    k: usize,
    is_movable: F,
) -> Vec<Cycle>
where
    F: Fn(EdgeIndex, EdgeIndex) -> bool,
{
    let mut queue = VecDeque::new();
    let mut cycles = Vec::new();

    // initialize
    for node in graph.node_indices() {
        queue.push_back((node, node, vec![]));
    }

    while let Some((v0, vn, edges)) = queue.pop_front() {
        // path v0 ---> vn
        let last_edge = edges.last().copied();

        // if graph has edge vn -> v0 this can be cycle
        for edge in graph.edges_connecting(vn, v0) {
            if last_edge.is_none() || is_movable(last_edge.unwrap(), edge.id()) {
                let mut cycle = edges.clone();
                cycle.push(edge.id());
                cycles.push(Cycle::new(cycle));
            }
        }

        // extend v0 ---> vn -> v
        if edges.len() < k - 1 {
            for edge in graph.edges_directed(vn, Direction::Outgoing) {
                if last_edge.is_none() || is_movable(last_edge.unwrap(), edge.id()) {
                    let v = edge.target();
                    let is_node_simple = edges.iter().all(|&edge| {
                        let (s, t) = graph.edge_endpoints(edge).unwrap();
                        s != v && t != v
                    });

                    if v.index() > v0.index() && is_node_simple {
                        let mut new_edges = edges.clone();
                        new_edges.push(edge.id());
                        queue.push_back((v0, v, new_edges));
                    }
                }
            }
        }
    }

    cycles
}

///
/// enumerate all cycles whose length is k
///
pub fn simple_k_cycles_from<N, E>(
    graph: &DiGraph<N, E>,
    k: usize,
    init_edge: EdgeIndex,
) -> Vec<Cycle> {
    simple_k_cycles_with_cond_from(graph, k, |_, _| true, init_edge)
}

///
/// enumerate all cycles whose length is k with `is_movable` condition
///
pub fn simple_k_cycles_with_cond_from<N, E, F>(
    graph: &DiGraph<N, E>,
    k: usize,
    is_movable: F,
    init_edge: EdgeIndex,
) -> Vec<Cycle>
where
    F: Fn(EdgeIndex, EdgeIndex) -> bool,
{
    //                             e1    e2     en
    // item in queue is a path `v0 -> v1 -> ... -> vn` (start node, end node, list of edges)
    let mut queue = VecDeque::new();
    let mut cycles = Vec::new();

    // initialize with single edge path
    //   init_edge
    // s --------> e
    let (start, end) = graph.edge_endpoints(init_edge).unwrap();
    queue.push_back((start, end, vec![init_edge]));

    // pick a path
    while let Some((v0, vn, edges)) = queue.pop_front() {
        println!("queue={:?} path={:?}{:?}{:?}", queue, v0, vn, edges);
        // path v0 ---> vn
        let last_edge = edges.last().copied();

        // Closing
        // if graph has edge vn -> v0 this path can be closed into a cycle
        for edge in graph.edges_connecting(vn, v0) {
            if last_edge.is_none() || is_movable(last_edge.unwrap(), edge.id()) {
                let mut cycle = edges.clone();
                cycle.push(edge.id());
                cycles.push(Cycle::new(cycle));
            }
        }

        // Extending
        // extend v0 ---> vn -> v
        if edges.len() < k - 1 {
            for edge in graph.edges_directed(vn, Direction::Outgoing) {
                if last_edge.is_none() || is_movable(last_edge.unwrap(), edge.id()) {
                    let v = edge.target();
                    let is_node_simple = edges.iter().all(|&edge| {
                        let (s, t) = graph.edge_endpoints(edge).unwrap();
                        s != v && t != v
                    });

                    if is_node_simple {
                        let mut new_edges = edges.clone();
                        new_edges.push(edge.id());
                        queue.push_back((v0, v, new_edges));
                    }
                }
            }
        }
    }

    cycles
}

///
/// find shortest cycle whose length <= k and considering `is_movable` condition
///
pub fn shortest_cycles_with_cond_from<N, E, F>(
    graph: &DiGraph<N, E>,
    k: usize,
    is_movable: F,
    init_edge: EdgeIndex,
) -> Option<Cycle>
where
    F: Fn(EdgeIndex, EdgeIndex) -> bool,
{
    //                             e1    e2     en
    // item in queue is a path `v0 -> v1 -> ... -> vn` (start node, end node, list of edges)
    let mut queue = VecDeque::new();

    // initialize with single edge path
    //   init_edge
    // s --------> e
    let (start, end) = graph.edge_endpoints(init_edge).unwrap();
    queue.push_back((start, end, vec![init_edge]));

    // pick a path
    while let Some((v0, vn, edges)) = queue.pop_front() {
        println!("queue={:?} path={:?}{:?}{:?}", queue, v0, vn, edges);
        // path v0 ---> vn
        let last_edge = edges.last().copied();

        // Closing
        // if graph has edge vn -> v0 this path can be closed into a cycle
        for edge in graph.edges_connecting(vn, v0) {
            if last_edge.is_none() || is_movable(last_edge.unwrap(), edge.id()) {
                let mut cycle = edges.clone();
                cycle.push(edge.id());
                return Some(Cycle::new(cycle));
            }
        }

        // Extending
        // extend v0 ---> vn -> v
        if edges.len() < k - 1 {
            for edge in graph.edges_directed(vn, Direction::Outgoing) {
                if last_edge.is_none() || is_movable(last_edge.unwrap(), edge.id()) {
                    let v = edge.target();
                    let is_node_simple = edges.iter().all(|&edge| {
                        let (s, t) = graph.edge_endpoints(edge).unwrap();
                        s != v && t != v
                    });

                    if is_node_simple {
                        let mut new_edges = edges.clone();
                        new_edges.push(edge.id());
                        queue.push_back((v0, v, new_edges));
                    }
                }
            }
        }
    }

    None
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{ei, ni};

    #[test]
    fn simple_cycles_test_1() {
        let g: DiGraph<(), ()> =
            DiGraph::from_edges(&[(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]);
        let cycles = simple_cycles_as_nodes(&g);
        println!("cycles={:?}", cycles);
        assert_eq!(
            cycles,
            vec![
                vec![ni(0)],
                vec![ni(0), ni(1), ni(2)],
                vec![ni(0), ni(2)],
                vec![ni(1), ni(2)],
                vec![ni(2)],
            ]
        );
    }
    #[test]
    fn simple_cycles_test_2() {
        let g: DiGraph<(), ()> =
            DiGraph::from_edges(&[(0, 2), (2, 1), (1, 3), (3, 2), (3, 4), (4, 5), (5, 1)]);
        let cycles = simple_cycles_as_nodes(&g);
        println!("cycles={:?}", cycles);
        assert_eq!(
            cycles,
            vec![vec![ni(1), ni(3), ni(2)], vec![ni(1), ni(3), ni(4), ni(5)],]
        );
    }
    #[test]
    fn simple_k_cycles_test_1() {
        let g: DiGraph<(), ()> =
            DiGraph::from_edges(&[(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]);
        let cycles = simple_k_cycles(&g, 2);
        println!("cycles={:?}", cycles);
        assert_eq!(
            cycles,
            vec![
                Cycle::from(&[0]),
                Cycle::from(&[6]),
                Cycle::from(&[2, 4]),
                Cycle::from(&[3, 5]),
            ]
        );
        let cycles = simple_k_cycles(&g, 10);
        println!("cycles={:?}", cycles);
        assert_eq!(
            cycles,
            vec![
                Cycle::from(&[0]),
                Cycle::from(&[6]),
                Cycle::from(&[2, 4]),
                Cycle::from(&[3, 5]),
                Cycle::from(&[1, 3, 4]),
            ]
        );
    }
    #[test]
    fn simple_k_cycles_test_2() {
        let g: DiGraph<(), ()> =
            DiGraph::from_edges(&[(0, 2), (2, 1), (1, 3), (3, 2), (3, 4), (4, 5), (5, 1)]);
        let cycles = simple_k_cycles(&g, 2);
        println!("cycles={:?}", cycles);
        assert_eq!(cycles.len(), 0);

        let cycles = simple_k_cycles(&g, 10);
        println!("cycles={:?}", cycles);
        assert_eq!(
            cycles,
            vec![Cycle::from(&[2, 3, 1]), Cycle::from(&[2, 4, 5, 6]),]
        );
    }
    #[test]
    fn simple_k_cycles_from_test_1() {
        let g: DiGraph<(), ()> = DiGraph::from_edges(&[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (2, 4),
            (4, 5),
            (4, 6),
            (6, 2),
        ]);
        let cycles = simple_k_cycles_from(&g, 10, ei(0));
        println!("cycles={:?}", cycles);
        assert_eq!(cycles, vec![Cycle::from(&[0, 1, 2, 3])]);

        let cycles = simple_k_cycles_from(&g, 3, ei(0));
        println!("cycles={:?}", cycles);
        assert_eq!(cycles.len(), 0);

        let cycles = simple_k_cycles_from(&g, 4, ei(1));
        println!("cycles={:?}", cycles);
        assert_eq!(cycles, vec![Cycle::from(&[1, 2, 3, 0])]);

        let cycles = simple_k_cycles_from(&g, 10, ei(6));
        println!("cycles={:?}", cycles);
        assert_eq!(cycles, vec![Cycle::from(&[6, 7, 4])]);

        let cycles = simple_k_cycles_from(&g, 10, ei(5));
        println!("cycles={:?}", cycles);
        assert_eq!(cycles.len(), 0);
    }
}
