//!
//! Cycle in graph
//!
use itertools::Itertools;
use petgraph::graph::EdgeIndex;
use std::cmp::Ordering;

///
/// Cycle (as a list of edges)
///
#[derive(Debug, Clone, PartialEq)]
pub struct Cycle(Vec<EdgeIndex>);

//
// Cycle
//
impl std::fmt::Display for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().map(|e| e.index()).join(","))
    }
}

impl Cycle {
    /// constructor from vec of edgeindex
    pub fn new(edges: Vec<EdgeIndex>) -> Cycle {
        Cycle(edges)
    }
    /// constructor from vec of usize slice
    pub fn from(indexes: &[usize]) -> Cycle {
        let edges = indexes.iter().map(|&i| EdgeIndex::new(i)).collect();
        Cycle::new(edges)
    }
    pub fn edges(&self) -> &[EdgeIndex] {
        &self.0
    }
    fn min_index(&self) -> usize {
        // find the index i (i=0,..,n-1) such that the suffix e[i:] is minimum
        let mut i0 = 0;
        let n = self.0.len();
        for i in 1..n {
            if let Ordering::Greater = cmp(&self.0, i0, i) {
                // i0 is (strictly) greater than i
                // that is the minimum candidate should be changed to i from i0
                i0 = i;
            }
        }
        i0
    }
    /// normalize the cycle
    /// so that a index vector will start in the minimum index.
    pub fn normalize(self) -> Cycle {
        let i = self.min_index();
        let mut new_cycle = self;
        new_cycle.0.rotate_left(i);
        new_cycle
    }
}

///
/// compare xs[i:] and xs[j:]
///
/// "`xs[i:]` is less/greater than `xs[j:]`?"
///
fn cmp<X: PartialOrd + Copy>(xs: &[X], i: usize, j: usize) -> Ordering {
    let n = xs.len();
    if i == j {
        return Ordering::Equal;
    }
    for k in 0..n {
        let xik = xs[(i + k) % n];
        let xjk = xs[(j + k) % n];
        if xik < xjk {
            return Ordering::Less;
        } else if xik > xjk {
            return Ordering::Greater;
        }
    }
    // all elements are the same
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_compare() {
        assert_eq!(cmp(&[3, 2, 4, 5, 1], 0, 1), Ordering::Greater);
        assert_eq!(cmp(&[3, 2, 4, 5, 1], 1, 0), Ordering::Less);
        assert_eq!(cmp(&[3, 2, 4, 5, 1], 0, 0), Ordering::Equal);
        assert_eq!(cmp(&[3, 2, 4, 3, 2, 4], 0, 3), Ordering::Equal);
    }

    #[test]
    fn cycle_normalize() {
        let c1 = Cycle::from(&[3, 2, 4, 5, 1]).normalize();
        println!("{:?}", c1);
        assert_eq!(c1, Cycle::from(&[1, 3, 2, 4, 5]));
        let c1 = Cycle::from(&[1, 1, 2, 1, 3]).normalize();
        println!("{:?}", c1);
        assert_eq!(c1, Cycle::from(&[1, 1, 2, 1, 3]));
        let c1 = Cycle::from(&[5, 1, 1, 2, 1, 3]).normalize();
        println!("{:?}", c1);
        assert_eq!(c1, Cycle::from(&[1, 1, 2, 1, 3, 5]));
        let c1 = Cycle::from(&[1]).normalize();
        println!("{:?}", c1);
        assert_eq!(c1, Cycle::from(&[1]));
    }
}
