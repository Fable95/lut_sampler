use fastnum::decimal::Decimal;
use tracing::instrument;

use crate::table_fill::greedy_fill::neg_power_two;

pub mod index_sampling;
pub mod greedy_fill;
pub mod many_fill;
pub mod pmfs;
pub mod export;

#[derive(Clone)]
pub struct TableParams<const S: usize> {
    pub table: Vec<LookupTable<u16>>,
    pub k: Vec<usize>,
    pub total_k: usize,
    pub l: usize,
    pub c_ber: u32,
    pub bound: usize,
    pub delta: Decimal<S>,
    pub limit: Decimal<S>,
}

impl<const S: usize> TableParams<S>{

    pub fn new(limit: Decimal<S>, bound: usize) -> Self{
        TableParams { table: vec![LookupTable::<u16>::new(&vec![1], 0u16)], k: vec![0], total_k: 0, l: 0, c_ber: 0, bound: bound, delta: Decimal::<S>::ONE, limit }
    }

    pub fn update(&mut self, table: Vec<LookupTable<u16>>, k: &Vec<usize>, l: usize, c_ber: u32, delta: Decimal<S>, cheap_bias: bool, optim_tables: bool) -> bool {
        let total_k = k.iter().sum();
        if delta > self.delta{
            return false;
        }
        if optim_tables && table.len() > self.table.len(){
            return false;
        }
        let stop_cond = if cheap_bias {
            total_k > self.total_k
        } else {
            total_k > self.total_k || c_ber > self.c_ber
        };

        if self.delta < self.limit && stop_cond{
            return false;
        }
        self.table = table;
        self.k = k.clone();
        self.total_k = total_k;
        self.l = l;
        self.c_ber = c_ber;
        self.delta = delta;
        return true;
    }
    pub fn all_set(tests: &[Self]) -> bool{
        tests.iter().fold(true, |acc, tc| acc && (tc.delta < tc.limit))
    }

    pub fn set(&self) -> bool{
        if self.delta > self.limit {
            return false;
        }
        true
    }
}

impl<const S: usize> std::fmt::Debug for TableParams<S>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} tables with k: {:?}, l: {}, p: 2^-{}, delta: {} ~ 2^{}, considered range: [0,{}]", 
            self.table.len(), self.k, self.l, self.c_ber, self.delta.to_scientific_notation(), neg_power_two(&self.delta),self.bound
        )
    }
}


#[derive(Debug, Clone)]
pub enum NDArray<T>{
    Leaf(T),
    Node(Vec<NDArray<T>>),
}

#[derive(Clone, Debug)]
pub struct LookupTable<T>{
    data: NDArray<T>,
    d: usize,
    k: Vec<usize>,
}

impl LookupTable<u16>{
    pub fn generate_pseudo_deterministic_cube<F>(k: &Vec<usize>, fun: F) -> Self
    where
    F: Fn(&[u16;3]) -> u16,
    {
        let mut data = Self::new(k, 0u16);
        let size_0 = 1<<k[0];
        let size_1 = 1<<k[1];
        let size_2 = 1<<k[2];
        for row in 0..size_0{
            for col in 0..size_1{
                for lay in 0..size_2{
                    let val = fun(&[row as u16, col as u16, lay as u16]);
                    data.set(&[row, col, lay], val);
                }
            }
        }
        data
    }
    pub fn generate_pseudo_deterministic_matrix<F>(k: &Vec<usize>, fun: F) -> Self
    where
    F: Fn(&[u16;2]) -> u16,
    {
        let mut data = Self::new(k, 0u16);
        let size_0 = 1<<k[0];
        let size_1 = 1<<k[1];
        for row in 0..size_0{
            for col in 0..size_1{
                let val = fun(&[row as u16, col as u16]);
                data.set(&[row, col], val);
            }
        }
        data
    }
}

impl<T: Copy> LookupTable<T>{
    #[instrument(name = "Setup ND_TABLE", skip_all)]
    pub fn new(k: &Vec<usize>, init: T) -> Self{
        let d = k.len();
        let data = Self::build_ndarray(d, k, init);
        LookupTable{data, d, k: k.clone()}
    }

    fn build_ndarray(depth: usize, branching: &Vec<usize>, init: T) -> NDArray<T> {
        assert!(branching.len() >= depth, "given branching parameters do not match depth");
        if depth == 0 {
            NDArray::Leaf(init)
        } else {
            // heap-allocated children
            let d = branching.len();
            let size = 1 << branching[d - depth];
            let mut children = Vec::with_capacity(size);
            for _ in 0..size {
                children.push(Self::build_ndarray(depth - 1, branching, init));
            }
            NDArray::Node(children)
        }
    }

    fn num_elements(&self) -> usize{
        1 << (self.k.iter().sum::<usize>())
    }

    pub fn root(&self) -> &NDArray<T> {
        &self.data
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        debug_assert_eq!(indices.len(), self.d, "Expected {} indices, got {}", self.d, indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            let bound = 1 << self.k[i];
            debug_assert!(idx < bound, "Index out of bounds at position {}: got {}, max {}", i, idx, bound - 1);
        }

        let mut node = &self.data;
        for &idx in indices {
            node = match node {
                NDArray::Node(children) => &children[idx],
                NDArray::Leaf(_) => panic!("Unexpected leaf at intermediate depth"),
            };
        }

        match node {
            NDArray::Leaf(val) => val,
            NDArray::Node(_) => panic!("Expected leaf at final depth"),
        }
    }
    pub fn set(&mut self, indices: &[usize], value: T) {
        debug_assert_eq!(indices.len(), self.d, "Expected {} indices, got {}", self.d, indices.len());

        for (i, &idx) in indices.iter().enumerate() {
            let bound = 1 << self.k[i];
            debug_assert!(idx < bound, "Index out of bounds at position {}: got {}, max {}", i, idx, bound - 1);
        }

        let mut node = &mut self.data;
        for &idx in &indices[..indices.len() - 1] {
            node = match node {
                NDArray::Node(children) => &mut children[idx],
                NDArray::Leaf(_) => panic!("Unexpected leaf at intermediate depth"),
            };
        }

        let last_idx = *indices.last().unwrap();
        match node {
            NDArray::Node(children) => {
                match &mut children[last_idx] {
                    NDArray::Leaf(val) => *val = value,
                    NDArray::Node(_) => panic!("Expected leaf at final depth"),
                }
            }
            NDArray::Leaf(_) => panic!("Unexpected leaf at depth {}", self.d - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use fastnum::{udec256, UD256};

    use super::*;

    #[test]
    fn test_lookup_cube_structure() {
        let k = vec![6,6,6]; // branching factor = 2^4 = 16
        let size: Vec<usize> = k.iter().map(|k| (1 << k) as usize).collect();
        let zero = 0u8;
        let offset_col = 8u8;
        let offset_lay = 64u8;
        let mut table = LookupTable::<u8>::new(&k, zero);
        for row in 0..size[0]{
            for col in 0..size[1]{
                for lay in 0..size[2]{
                    let indices = [row, col, lay];
                    let val = offset_col.wrapping_mul(col as u8).wrapping_add(offset_lay.wrapping_mul(lay as u8));
                    table.set(&indices, val);
                    
                }
            }
        }
        match table.root() {
            NDArray::Node(level1) => {
                assert_eq!(level1.len(), size[0]);
                for l1 in level1 {
                    match l1 {
                        NDArray::Node(level2) => {
                            assert_eq!(level2.len(), size[1]);
                            for (col, l2) in level2.iter().enumerate() {
                                match l2 {
                                    NDArray::Node(level3) => {
                                        assert_eq!(level3.len(), size[2]);
                                        for (lay, leaf) in level3.iter().enumerate() {
                                            let expected = offset_col.wrapping_mul(col as u8).wrapping_add(offset_lay.wrapping_mul(lay as u8));
                                            assert!(
                                                matches!(leaf, NDArray::Leaf(val) if *val == expected),
                                                "Leaf was not {:?}: got {:?}",
                                                expected, leaf
                                            );
                                        }
                                    }
                                    _ => panic!("Expected node at depth 2"),
                                }
                            }
                        }
                        _ => panic!("Expected node at depth 1"),
                    }
                }
            }
            _ => panic!("Expected node at root"),
        }
    }

    #[test]
    fn test_lookup_table_structure() {
        let k = vec![24]; // branching factor = 2^4 = 16
        let size: Vec<usize> = k.iter().map(|k| (1 << k) as usize).collect();
        let zero = udec256!(0);
        let offset_row = udec256!(0.00000005960464477539063);
        let mut table = LookupTable::<UD256>::new(&k, zero);
        for row in 0..size[0]{
            let indices = [row];
            let val = offset_row * row;
            table.set(&indices, val);
        }
        match table.root() {
            NDArray::Node(level1) => {
                assert_eq!(level1.len(), size[0]);
                for (element, leaf) in level1.iter().enumerate() {
                    let expected = offset_row * element;
                    assert!(
                        matches!(leaf, NDArray::Leaf(val) if *val == expected),
                        "Leaf was not {:?}: got {:?}",
                        expected, leaf
                    );
                }
            }
            _ => panic!("Expected node at depth 1"),
        }
            
            
            
    }
}