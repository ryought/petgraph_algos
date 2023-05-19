//!
//!
//!
use super::common::FloatWeight;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, VisitMap, Visitable};

///
///
///
pub fn find_negative_cycle<N, E: FloatWeight>(
    g: &DiGraph<N, E>,
    source: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    let (mut dist, mut pred) = bellman_ford(g, source);
    let ix = |i: NodeIndex| i.index();

    //
    // The Bellman-ford relaxing function above checks all shortest paths whose edge count <= |V|-1
    // If the shortest paths can be relaxed more, there should be a negative cycle.
    //
    for edge in g.edge_references() {
        //    w
        // u ---> v
        let u = edge.source();
        let v = edge.target();
        let w = edge.weight().float_weight();
        if dist[ix(u)] + w + E::epsilon() < dist[ix(v)] {
            // A new shortest path from source to v was found
            // ending with this edge u->v
            //
            // Length of this path is |V|, so there should be a duplicate node (that was visited
            // twice by this path). `traceback` will find the cyclic subpath.
            dist[ix(v)] = dist[ix(u)] + w;
            pred[ix(v)] = Some(u);
            return Some(traceback(g, &pred, v));
        }
    }

    None
}

///
/// Find a cycle in shortest path from `source` to `target`
///
/// by using Bellman-Ford's pred (shortest path tree) on graph `g`.
///
pub fn traceback<N, E>(
    g: &DiGraph<N, E>,
    pred: &[Option<NodeIndex>],
    target: NodeIndex,
) -> Vec<NodeIndex> {
    let mut path = Vec::new();
    let mut node = target;
    let mut visited = g.visit_map();
    path.push(node);
    visited.visit(node);

    loop {
        println!("node={}", node.index());
        node = pred[node.index()].expect("no pred!");

        // loop detected
        if visited.is_visited(&node) {
            let pos = path
                .iter()
                .position(|&p| p == node)
                .expect("we should always have a position");
            path = path[pos..path.len()].to_vec();
            break;
        }

        path.push(node);
        visited.visit(node);
    }

    path.reverse();
    path
}

///
/// Calculate distances from source to node `dist[node]` and predecessor node list `pred[node]`
///
pub fn bellman_ford<N, E: FloatWeight>(
    g: &DiGraph<N, E>,
    source: NodeIndex,
) -> (Vec<f64>, Vec<Option<NodeIndex>>) {
    // init
    let mut dist = vec![f64::INFINITY; g.node_count()];
    let mut pred = vec![None; g.node_count()];
    let ix = |i: NodeIndex| i.index();
    dist[ix(source)] = 0.0;

    // relax
    for i in 1..g.node_count() {
        let mut dist_new = dist.clone();
        for edge in g.edge_references() {
            //    w
            // u ---> v
            let u = edge.source();
            let v = edge.target();
            let w = edge.weight().float_weight();
            if dist[ix(u)] + w + E::epsilon() < dist[ix(v)] {
                dist_new[ix(v)] = dist[ix(u)] + w;
                pred[ix(v)] = Some(u);
            }
        }
        dist = dist_new;
    }

    (dist, pred)
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bellman_ford_test_01() {
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let s = g.add_node(());
        let t = g.add_node(());
        let x = g.add_node(());
        let y = g.add_node(());
        let z = g.add_node(());
        g.add_edge(s, t, 6.0);
        g.add_edge(s, y, 7.0);
        g.add_edge(t, x, 5.0);
        g.add_edge(t, y, 8.0);
        g.add_edge(t, z, -4.0);
        g.add_edge(x, t, -2.0);
        g.add_edge(y, x, -3.0);
        g.add_edge(y, z, 9.0);
        g.add_edge(z, s, 2.0);
        g.add_edge(z, x, 7.0);
        let (dist, pred) = bellman_ford(&g, s);
        println!("{:?}", dist);
        println!("{:?}", pred);
        assert_eq!(dist, vec![0.0, 2.0, 4.0, 7.0, -2.0]);
        assert!(find_negative_cycle(&g, s).is_none());
    }

    #[test]
    fn bellman_ford_test_02() {
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let s = g.add_node(());
        let x = g.add_node(());
        let y = g.add_node(());
        g.add_edge(s, x, 5.0);
        g.add_edge(s, y, 100.0);
        g.add_edge(x, y, 10.0);
        let (dist, pred) = bellman_ford(&g, s);
        println!("{:?}", dist);
        println!("{:?}", pred);
        assert_eq!(dist, vec![0.0, 5.0, 15.0]);
        assert!(find_negative_cycle(&g, s).is_none());
    }

    #[test]
    fn bellman_ford_test_03() {
        // Introduction to algorithm p646 Figure24.1
        let mut graph: DiGraph<(), f64> = DiGraph::new();
        let s = graph.add_node(());
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());
        let f = graph.add_node(());
        let g = graph.add_node(());
        let h = graph.add_node(());
        let i = graph.add_node(());
        let j = graph.add_node(());

        graph.add_edge(s, a, 3.0);
        graph.add_edge(s, c, 5.0);
        graph.add_edge(s, e, 2.0);
        graph.add_edge(a, b, -4.0);
        graph.add_edge(b, g, 4.0);
        graph.add_edge(c, d, 6.0);
        graph.add_edge(d, c, -3.0);
        graph.add_edge(d, g, 8.0);
        graph.add_edge(e, f, 3.0);
        graph.add_edge(f, e, -6.0);
        graph.add_edge(f, g, 7.0);
        graph.add_edge(h, i, 2.0);
        graph.add_edge(i, j, 3.0);
        graph.add_edge(j, h, -8.0);

        let (dist, pred) = bellman_ford(&graph, s);
        println!("{:?}", dist);
        println!("{:?}", pred);
        let inf = f64::INFINITY;
        assert_eq!(
            dist,
            vec![0.0, 3.0, -1.0, 5.0, 11.0, -10.0, -7.0, 3.0, inf, inf, inf]
        );

        let cycle = find_negative_cycle(&graph, s);
        println!("{:?}", cycle);
        assert_eq!(cycle, Some(vec![f, e]));
    }

    #[test]
    fn bellman_ford_test_04() {
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let s = g.add_node(());
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        let d = g.add_node(());
        let e = g.add_node(());
        g.add_edge(s, a, 1.0);
        g.add_edge(s, c, 10.0);
        g.add_edge(a, b, 9.0);
        g.add_edge(a, c, 2.0);
        g.add_edge(b, e, -20.0);
        g.add_edge(c, d, 5.0);
        g.add_edge(d, a, 1.0);
        g.add_edge(e, d, 1.0);
        let (dist, pred) = bellman_ford(&g, s);
        println!("{:?}", dist);
        println!("{:?}", pred);
        // assert_eq!(dist, vec![0.0, 5.0, 15.0]);
        let cycle = find_negative_cycle(&g, s);
        println!("{:?}", cycle);
        assert_eq!(cycle, Some(vec![e, d, a, b]));
    }

    #[test]
    fn bellman_ford_test_05() {
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let v0 = g.add_node(());
        let v1 = g.add_node(());
        let v2 = g.add_node(());
        g.add_edge(v0, v1, -9.0);
        g.add_edge(v1, v2, -9.0);
        g.add_edge(v2, v0, -9.0);
        let (dist, pred) = bellman_ford(&g, v0);
        println!("{:?}", dist);
        println!("{:?}", pred);
        // assert_eq!(dist, vec![0.0, 5.0, 15.0]);
        let cycle = find_negative_cycle(&g, v0);
        println!("{:?}", cycle);
        assert_eq!(cycle, Some(vec![v1, v2, v0]));
    }

    #[test]
    fn bellman_ford_test_06() {
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let v = g.add_node(());
        g.add_edge(v, v, -9.0);
        let (dist, pred) = bellman_ford(&g, v);
        println!("{:?}", dist);
        println!("{:?}", pred);
        let cycle = find_negative_cycle(&g, v);
        println!("{:?}", cycle);
        assert_eq!(cycle, Some(vec![v]));
    }

    #[test]
    fn bellman_ford_test_07() {
        let mut g: DiGraph<(), f64> = DiGraph::new();
        let v0 = g.add_node(());
        let v1 = g.add_node(());
        g.add_edge(v0, v1, -9.0);
        g.add_edge(v1, v0, 8.0);
        let (dist, pred) = bellman_ford(&g, v0);
        println!("{:?}", dist);
        println!("{:?}", pred);
        let cycle = find_negative_cycle(&g, v0);
        println!("{:?}", cycle);
        assert_eq!(cycle, Some(vec![v1, v0]));
    }

    #[test]
    fn bellman_ford_negative_cycle_complex_case() {
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
        let (dist, pred) = bellman_ford(&g, NodeIndex::new(0));
        println!("{:?}", dist);
        println!("{:?}", pred);
        let cycle = find_negative_cycle(&g, NodeIndex::new(0));
        println!("{:?}", cycle);
        assert_eq!(cycle, Some(vec![NodeIndex::new(25), NodeIndex::new(14)]));
    }
}
