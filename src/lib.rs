//!
//! Graph algorithms for petgraph::Graph
//!
//! # Algorithms
//!
//! * bellman_ford: Bellman Ford using FloatWeight
//! * floyd: Floyd using FloatWeight
//! * cycle_enumeration: simple cycle enumeration
//!
//! # Wrappers
//!
//! * iterators
//!
pub mod bellman_ford;
pub mod common;
pub mod cycle;
pub mod cycle_enumeration;
pub mod floyd;
pub mod iterators;
pub mod min_mean_weight_cycle;
