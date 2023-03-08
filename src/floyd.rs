//!
//! Floyd-Warshall algorithm
//! to find minimum weight cycle
//!
use super::common::FloatWeight;
use petgraph::prelude::*;
use petgraph::visit::{VisitMap, Visitable};

///
/// A variant of Floyd Warshall algorithm
/// to find a shortest path distance of all pairs of nodes, without edge repetitions.
///
pub fn floyd_warshall<N, E: FloatWeight>(graph: &DiGraph<N, E>) -> Vec<Vec<f64>> {
    let n_nodes = graph.node_count();
    let n_edges = graph.edge_count();
    let mut dist0 = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
    let mut dist1 = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
    let ix = |node: NodeIndex| node.index();

    // (1-a) init
    for v in graph.node_indices() {
        dist1[ix(v)][ix(v)] = 0.0;
    }

    for (k, edge) in graph.edge_references().enumerate() {
        let s = edge.source();
        let t = edge.target();
        let w = edge.weight().float_weight();

        println!("k={} s={} t={} w={}", k, s.index(), t.index(), w);

        // fill dist1
        if k == 0 {
            // (1-b) init
            // d[s,t] = w_st
            dist1[ix(s)][ix(t)] = w;
        } else {
            // (2) step
            for i in graph.node_indices() {
                for j in graph.node_indices() {
                    // min(d[i,j], d[i,s]+w[s,t]+d[t,j])
                    dist1[ix(i)][ix(j)] =
                        dist0[ix(i)][ix(j)].min(dist0[ix(i)][ix(s)] + w + dist0[ix(t)][ix(j)]);
                }
            }
        }

        println!("dist0={:?}", dist0);
        println!("dist1={:?}", dist1);

        // store dist1 as dist0 for next iteration
        dist0 = dist1;
        dist1 = vec![vec![f64::INFINITY; n_nodes]; n_nodes];
    }

    dist0
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floyd_warshall_01() {
        //
        // graph used as an example of petgraph floydWarshall
        //
        let mut g: DiGraph<(), f64> = DiGraph::new();
        g.extend_with_edges(&[
            (0, 1, 1.0),
            (0, 2, 4.0),
            (0, 3, 10.0),
            (1, 2, 2.0),
            (1, 3, 2.0),
            (2, 3, 2.0),
        ]);
        let dist = floyd_warshall(&g);
        println!("{:?}", dist);
        let inf = f64::INFINITY;
        assert_eq!(
            dist,
            vec![
                vec![0.0, 1.0, 3.0, 3.0],
                vec![inf, 0.0, 2.0, 2.0],
                vec![inf, inf, 0.0, 2.0],
                vec![inf, inf, inf, 0.0],
            ]
        );
    }
    #[test]
    fn floyd_warshall_02() {
        //
        // graph containing negative loop
        //
        let mut g: DiGraph<(), f64> = DiGraph::new();
        g.extend_with_edges(&[
            (0, 1, 2.0),
            (0, 4, -3.0),
            (1, 2, -1.0),
            (2, 3, -1.0),
            (3, 1, -1.0),
            (4, 3, 3.0),
        ]);
        let dist = floyd_warshall(&g);
        println!("{:?}", dist);
    }
    #[test]
    fn floyd_warshall_03() {
        //
        // graph containing negative loop
        //
        let mut g: DiGraph<(), f64> = DiGraph::new();
        g.extend_with_edges(&[(0, 1, -2.0), (1, 2, -3.0), (2, 0, -4.0)]);
        let dist = floyd_warshall(&g);
        println!("{:?}", dist);
    }
}
