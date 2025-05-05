use tracing::instrument;

pub mod index_sampling;
pub mod greedy_fill;
pub mod pmfs;
pub mod export;

#[derive(Debug)]
pub enum NDArray<T>{
    Leaf(T),
    Node(Vec<NDArray<T>>),
}

#[derive(Debug)]
pub struct LookupTable<T>{
    data: NDArray<T>,
    d: usize,
    k: usize,
}

impl LookupTable<u8>{
    pub fn generate_pseudo_deterministic<F>(d: usize, k: usize, fun: F) -> Self
    where
    F: Fn(u8, u8, u8) -> u8,
    {
        let size = 1 << k;
        let mut data = Self::new(d, k, 0u8);

        for row in 0..size{
            for col in 0..size{
                for lay in 0..size{
                    let val = fun(row as u8, col as u8, lay as u8);
                    data.set(&[row, col, lay], val);
                }
            }
        }
        data
    }
}

impl<T: Copy> LookupTable<T>{
    #[instrument(name = "Setup ND_TABLE", skip_all)]
    pub fn new(d: usize, k: usize, init: T) -> Self{
        let size = 1 << k;
        let data = Self::build_ndarray(d, size, init);
        LookupTable{data, d, k}
    }

    fn build_ndarray(depth: usize, branching: usize, init: T) -> NDArray<T> {
        if depth == 0 {
            NDArray::Leaf(init)
        } else {
            // heap-allocated children
            let mut children = Vec::with_capacity(branching);
            for _ in 0..branching {
                children.push(Self::build_ndarray(depth - 1, branching, init));
            }
            NDArray::Node(children)
        }
    }

    fn num_elements(&self) -> usize{
        1 << (self.k * self.d)
    }

    pub fn root(&self) -> &NDArray<T> {
        &self.data
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        debug_assert_eq!(indices.len(), self.d, "Expected {} indices, got {}", self.d, indices.len());
        let bound = 1 << self.k;
        for (i, &idx) in indices.iter().enumerate() {
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
        let bound = 1 << self.k;
        for (i, &idx) in indices.iter().enumerate() {
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
        let d = 3;
        let k = 6; // branching factor = 2^4 = 16
        let size = (1 << k) as usize;
        let zero = 0u8;
        let offset_col = 8u8;
        let offset_lay = 64u8;
        let mut table = LookupTable::<u8>::new(d, k, zero);
        for row in 0..size{
            for col in 0..size{
                for lay in 0..size{
                    let indices = [row, col, lay];
                    let val = offset_col.wrapping_mul(col as u8).wrapping_add(offset_lay.wrapping_mul(lay as u8));
                    table.set(&indices, val);
                    
                }
            }
        }
        match table.root() {
            NDArray::Node(level1) => {
                assert_eq!(level1.len(), size);
                for l1 in level1 {
                    match l1 {
                        NDArray::Node(level2) => {
                            assert_eq!(level2.len(), size);
                            for (col, l2) in level2.iter().enumerate() {
                                match l2 {
                                    NDArray::Node(level3) => {
                                        assert_eq!(level3.len(), size);
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
        let d = 1;
        let k = 24; // branching factor = 2^4 = 16
        let size = 1 << k;
        let zero = udec256!(0);
        let offset_row = udec256!(0.00000005960464477539063);
        let mut table = LookupTable::<UD256>::new(d, k, zero);
        for row in 0..size{
            let indices = [row];
            let val = offset_row * row;
            table.set(&indices, val);
        }
        match table.root() {
            NDArray::Node(level1) => {
                assert_eq!(level1.len(), size);
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