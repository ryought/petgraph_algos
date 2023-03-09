//!
//! Wrapper of petgraph::Graph iterators
//!
//! * nodes()
//! * edges()
//! * childs()
//! * parents()
//!
use petgraph::graph::{DiGraph, EdgeIndex, EdgeReferences, Edges, NodeIndex, NodeReferences};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use petgraph::{Directed, Direction};

///
/// Iterator struct for `nodes()`
///
/// implements Iterator whose item is
/// `(node: NodeIndex, node_weight: &N)`
///
/// wrapper of DiGraph::node_references()
///
pub struct NodesIterator<'a, N: 'a> {
    nodes: NodeReferences<'a, N>,
}

impl<'a, N> NodesIterator<'a, N> {
    ///
    /// Create NodesIterator from DiGraph
    ///
    pub fn new<E>(graph: &'a DiGraph<N, E>) -> Self {
        NodesIterator {
            nodes: graph.node_references(),
        }
    }
}

impl<'a, N> Iterator for NodesIterator<'a, N> {
    type Item = (NodeIndex, &'a N);
    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.next()
    }
}

///
/// Iterator struct for `edges()`
///
/// implements Iterator whose item is
/// `(edge: EdgeIndex, source: NodeIndex, target: NodeIndex, edge_weight: &E)`
///
/// wrapper of DiGraph::edge_references()
///
pub struct EdgesIterator<'a, E: 'a> {
    edges: EdgeReferences<'a, E>,
}

impl<'a, E> EdgesIterator<'a, E> {
    ///
    /// Create EdgesIterator from the reference of DiGraph
    ///
    pub fn new<N>(graph: &'a DiGraph<N, E>) -> Self {
        EdgesIterator {
            edges: graph.edge_references(),
        }
    }
}

impl<'a, E> Iterator for EdgesIterator<'a, E> {
    type Item = (EdgeIndex, NodeIndex, NodeIndex, &'a E);
    fn next(&mut self) -> Option<Self::Item> {
        // extract edge reference
        match self.edges.next() {
            Some(er) => Some((er.id(), er.source(), er.target(), er.weight())),
            None => None,
        }
    }
}

///
/// Iterator for `childs()`
///
/// implements Iterator whose item is
/// `(edge: EdgeIndex, child: NodeIndex, edge_weight: &E)`
///
/// wrapper of DiGraph::edges_directed(_, Outgoing)
///
pub struct ChildEdges<'a, E: 'a> {
    edges: Edges<'a, E, Directed>,
}

impl<'a, E> ChildEdges<'a, E> {
    ///
    /// Create ChildEdges from the reference of DiGraph
    ///
    pub fn new<N>(graph: &'a DiGraph<N, E>, node: NodeIndex) -> Self {
        ChildEdges {
            edges: graph.edges_directed(node, Direction::Outgoing),
        }
    }
}

impl<'a, E> Iterator for ChildEdges<'a, E> {
    type Item = (EdgeIndex, NodeIndex, &'a E);
    fn next(&mut self) -> Option<Self::Item> {
        // edge reference
        match self.edges.next() {
            // er.source() = the given node
            // er.target() = child
            Some(er) => Some((er.id(), er.target(), er.weight())),
            None => None,
        }
    }
}

///
/// Iterator for `parents()`
///
/// implements Iterator whose item is
/// `(edge: EdgeIndex, parent: NodeIndex, edge_weight: &E)`
///
/// wrapper of DiGraph::edges_directed(_, Incoming)
///
pub struct ParentEdges<'a, E: 'a> {
    pub edges: Edges<'a, E, Directed>,
}

impl<'a, E> ParentEdges<'a, E> {
    ///
    /// Create ParentEdges from the reference of DiGraph
    ///
    pub fn new<N>(graph: &'a DiGraph<N, E>, node: NodeIndex) -> Self {
        ParentEdges {
            edges: graph.edges_directed(node, Direction::Incoming),
        }
    }
}

impl<'a, E> Iterator for ParentEdges<'a, E> {
    type Item = (EdgeIndex, NodeIndex, &'a E);
    fn next(&mut self) -> Option<Self::Item> {
        // edge reference
        match self.edges.next() {
            // er.source() = parent
            // er.target() = the given node
            Some(er) => Some((er.id(), er.source(), er.weight())),
            None => None,
        }
    }
}
