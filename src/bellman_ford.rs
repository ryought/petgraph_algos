//! Bellman-Ford algorithms for general EdgeWeight
//!
//! Copied from https://github.com/petgraph/petgraph/blob/master/src/algo/bellman_ford.rs
//! Modified to use generalized weight.
//! It works general EdgeWeight satisfying FloatWeight.

use super::common::FloatWeight;
use petgraph::algo::NegativeCycle;
use petgraph::prelude::*;
use petgraph::visit::{
    IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable, VisitMap, Visitable,
};

#[derive(Debug, Clone)]
pub struct Paths<NodeId, EdgeWeight> {
    pub distances: Vec<EdgeWeight>,
    pub predecessors: Vec<Option<NodeId>>,
}

/// \[Generic\] Compute shortest paths from node `source` to all other.
///
/// Using the [Bellman–Ford algorithm][bf]; negative edge costs are
/// permitted, but the graph must not have a cycle of negative weights
/// (in that case it will return an error).
///
/// On success, return one vec with path costs, and another one which points
/// out the predecessor of a node along a shortest path. The vectors
/// are indexed by the graph's node indices.
///
/// [bf]: https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
///
/// # Example
/// ```rust
/// use petgraph::Graph;
/// use petgraph::algo::bellman_ford;
/// use petgraph::prelude::*;
///
/// let mut g = Graph::new();
/// let a = g.add_node(()); // node with no weight
/// let b = g.add_node(());
/// let c = g.add_node(());
/// let d = g.add_node(());
/// let e = g.add_node(());
/// let f = g.add_node(());
/// g.extend_with_edges(&[
///     (0, 1, 2.0),
///     (0, 3, 4.0),
///     (1, 2, 1.0),
///     (1, 5, 7.0),
///     (2, 4, 5.0),
///     (4, 5, 1.0),
///     (3, 4, 1.0),
/// ]);
///
/// // Graph represented with the weight of each edge
/// //
/// //     2       1
/// // a ----- b ----- c
/// // | 4     | 7     |
/// // d       f       | 5
/// // | 1     | 1     |
/// // \------ e ------/
///
/// let path = bellman_ford(&g, a);
/// assert!(path.is_ok());
/// let path = path.unwrap();
/// assert_eq!(path.distances, vec![    0.0,     2.0,    3.0,    4.0,     5.0,     6.0]);
/// assert_eq!(path.predecessors, vec![None, Some(a),Some(b),Some(a), Some(d), Some(e)]);
///
/// // Node f (indice 5) can be reach from a with a path costing 6.
/// // Predecessor of f is Some(e) which predecessor is Some(d) which predecessor is Some(a).
/// // Thus the path from a to f is a <-> d <-> e <-> f
///
/// let graph_with_neg_cycle = Graph::<(), f32, Undirected>::from_edges(&[
///         (0, 1, -2.0),
///         (0, 3, -4.0),
///         (1, 2, -1.0),
///         (1, 5, -25.0),
///         (2, 4, -5.0),
///         (4, 5, -25.0),
///         (3, 4, -1.0),
/// ]);
///
/// assert!(bellman_ford(&graph_with_neg_cycle, NodeIndex::new(0)).is_err());
/// ```
pub fn bellman_ford<G>(g: G, source: G::NodeId) -> Result<Paths<G::NodeId, f64>, NegativeCycle>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges + NodeIndexable,
    G::EdgeWeight: FloatWeight,
{
    let ix = |i| g.to_index(i);

    // Step 1 and Step 2: initialize and relax
    let (distances, predecessors) = bellman_ford_initialize_relax(g, source);

    // Step 3: check for negative weight cycle
    for i in g.node_identifiers() {
        for edge in g.edges(i) {
            let j = edge.target();
            let w = edge.weight().float_weight();
            if distances[ix(i)] + w + G::EdgeWeight::epsilon() < distances[ix(j)] {
                return Err(NegativeCycle(()));
            }
        }
    }

    Ok(Paths {
        distances,
        predecessors,
    })
}

/// \[Generic\] Find the path of a negative cycle reachable from node `source`.
///
/// Using the [find_negative_cycle][nc]; will search the Graph for negative cycles using
/// [Bellman–Ford algorithm][bf]. If no negative cycle is found the function will return `None`.
///
/// If a negative cycle is found from source, return one vec with a path of `NodeId`s.
///
/// The time complexity of this algorithm should be the same as the Bellman-Ford (O(|V|·|E|)).
///
/// [nc]: https://blogs.asarkar.com/assets/docs/algorithms-curated/Negative-Weight%20Cycle%20Algorithms%20-%20Huang.pdf
/// [bf]: https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
///
/// # Example
/// ```rust
/// use petgraph::Graph;
/// use petgraph::algo::find_negative_cycle;
/// use petgraph::prelude::*;
///
/// let graph_with_neg_cycle = Graph::<(), f32, Directed>::from_edges(&[
///         (0, 1, 1.),
///         (0, 2, 1.),
///         (0, 3, 1.),
///         (1, 3, 1.),
///         (2, 1, 1.),
///         (3, 2, -3.),
/// ]);
///
/// let path = find_negative_cycle(&graph_with_neg_cycle, NodeIndex::new(0));
/// assert_eq!(
///     path,
///     Some([NodeIndex::new(1), NodeIndex::new(3), NodeIndex::new(2)].to_vec())
/// );
/// ```
pub fn find_negative_cycle<G>(g: G, source: G::NodeId) -> Option<Vec<G::NodeId>>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges + NodeIndexable + Visitable,
    G::EdgeWeight: FloatWeight,
{
    let ix = |i| g.to_index(i);
    let mut path = Vec::<G::NodeId>::new();

    // Step 1: initialize and relax
    let (distance, predecessor) = bellman_ford_initialize_relax(g, source);

    // Step 2: Check for negative weight cycle
    'outer: for i in g.node_identifiers() {
        for edge in g.edges(i) {
            println!(
                "checking edge {}->{} weight={}",
                g.to_index(edge.source()),
                g.to_index(edge.target()),
                edge.weight().float_weight(),
            );
            let j = edge.target();
            let w = edge.weight().float_weight();
            if distance[ix(i)] + w + G::EdgeWeight::epsilon() < distance[ix(j)] {
                println!(
                    "negative cycle found i={} j={} w={} d[i]={} d[j]={} L={} R={}\n{:?}",
                    ix(i),
                    ix(j),
                    w,
                    distance[ix(i)],
                    distance[ix(j)],
                    distance[ix(i)] + w + G::EdgeWeight::epsilon(),
                    distance[ix(j)],
                    distance,
                );
                // Step 3: negative cycle found
                let start = j;
                let mut node = start;
                let mut visited = g.visit_map();
                // Go backward in the predecessor chain
                loop {
                    println!("node={}", ix(node));
                    let ancestor = match predecessor[ix(node)] {
                        Some(predecessor_node) => {
                            println!("pred");
                            predecessor_node
                        }
                        None => {
                            println!("no pred");
                            panic!("no pred")
                            // node // no predecessor, self cycle
                        }
                    };
                    println!("ancestor={}", ix(ancestor));
                    // We have only 2 ways to find the cycle and break the loop:
                    // 1. start is reached
                    if ancestor == start {
                        path.push(ancestor);
                        break;
                    }
                    // 2. some node was reached twice
                    else if visited.is_visited(&ancestor) {
                        // Drop any node in path that is before the first ancestor
                        let pos = path
                            .iter()
                            .position(|&p| p == ancestor)
                            .expect("we should always have a position");
                        path = path[pos..path.len()].to_vec();

                        break;
                    }

                    // None of the above, some middle path node
                    path.push(ancestor);
                    visited.visit(ancestor);
                    node = ancestor;
                }
                // We are done here
                break 'outer;
            }
        }
    }
    if !path.is_empty() {
        // Users will probably need to follow the path of the negative cycle
        // so it should be in the reverse order than it was found by the algorithm.
        path.reverse();
        Some(path)
    } else {
        None
    }
}

// Perform Step 1 and Step 2 of the Bellman-Ford algorithm.
#[inline(always)]
fn bellman_ford_initialize_relax<G>(g: G, source: G::NodeId) -> (Vec<f64>, Vec<Option<G::NodeId>>)
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges + NodeIndexable,
    G::EdgeWeight: FloatWeight,
{
    // Step 1: initialize graph
    let mut predecessor = vec![None; g.node_bound()];
    let mut distance = vec![f64::INFINITY; g.node_bound()];
    let ix = |i| g.to_index(i);
    distance[ix(source)] = 0.0;

    // Step 2: relax edges repeatedly
    for _ in 1..g.node_count() {
        let mut did_update = false;
        for i in g.node_identifiers() {
            for edge in g.edges(i) {
                let j = edge.target();
                let w = edge.weight().float_weight();
                if distance[ix(i)] + w + G::EdgeWeight::epsilon() < distance[ix(j)] {
                    println!(
                        "updated\ti={}\tj={}\tw={}\td[i={}]={}\td[j={}]={}",
                        ix(i),
                        ix(j),
                        w,
                        ix(i),
                        distance[ix(i)],
                        ix(j),
                        distance[ix(j)]
                    );
                    distance[ix(j)] = distance[ix(i)] + w;
                    predecessor[ix(j)] = Some(i);
                    did_update = true;
                }
            }
        }
        if !did_update {
            break;
        }
    }
    (distance, predecessor)
}

//
// test
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bellman_ford_custom() {
        let g = DiGraph::<(), f64>::from_edges(&[
            (0, 1, 1000000.1),
            (0, 2, 10.001),
            (1, 2, 0.0001),
            (2, 1, -0.0001),
        ]);
        let path = find_negative_cycle(&g, NodeIndex::new(0));
        println!("{:?}", path);

        let g = DiGraph::<(), f64>::from_edges(&[
            (0, 1, 1000000.1),
            (0, 2, 10.001),
            (1, 2, 0.0002),
            (2, 1, -0.0005),
        ]);
        let path = find_negative_cycle(&g, NodeIndex::new(0));
        println!("{:?}", path);
    }

    #[test]
    fn orig_negative_cycle_complex_case() {
        let mut g: DiGraph<(), f64> = DiGraph::from_edges(&[
            (0, 17, 1.0),
            (17, 0, 1.0),
            (17, 3, 1.0),
            (3, 17, 1.0),
            (3, 2, 1.0),
            (2, 3, 1.0),
            (2, 31, 113.0),
            (31, 2, 138.0),
            (31, 7, 5.0),
            (7, 31, 50.0),
            (7, 12, 0.0),
            (12, 7, 1.0),
            (12, 10, 1.0),
            (10, 12, 54.0),
            (10, 26, 1165.0),
            (26, 10, 712.0),
            (26, 30, 163.0),
            (30, 26, 74.0),
            (30, 16, 604.0),
            (16, 30, 221.0),
            (16, 14, 581.0),
            (14, 16, -171.0),
            (14, 25, -35.0),
            (25, 14, -139.0),
            (25, 32, 0.0),
            (32, 25, -0.0),
        ]);
        println!("{:?}", petgraph::dot::Dot::with_config(&g, &[]));
        let (dist, pred) = bellman_ford_initialize_relax(&g, NodeIndex::new(0));
        println!("{:?}", dist);
        println!("{:?}", pred);
        for i in 0..g.node_count() {
            println!("i={} d[i]={} p[i]={:?}", i, dist[i], pred[i]);
        }
        let c = find_negative_cycle(&g, NodeIndex::new(0));
        // println!("{:?}", c);
    }
}
