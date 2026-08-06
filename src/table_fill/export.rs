// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
use core::panic;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use fastnum::decimal::Decimal;

use super::{LookupTable, NDArray, greedy_fill::neg_power_two};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedType{
    U8,
    U16,
    BitShare
}

impl EmbeddedType{
    pub fn to_str(&self) -> &str {
        match self {
            EmbeddedType::U8 => "u8",
            EmbeddedType::U16 => "u16",
            EmbeddedType::BitShare => "BitShare",
        }
    }

    pub fn ratio(&self) -> usize {
        match self {
            EmbeddedType::U8 => 8,
            EmbeddedType::U16 => 4,
            EmbeddedType::BitShare => 8,
        }
    }

    pub fn write_compressed_element_to_u64(&self, values: &Vec<u16>) -> String {
        match self {
            EmbeddedType::U8 => write_compressed_element_u8_u64(values),
            EmbeddedType::U16 => write_compressed_element_u16_u64(values),
            EmbeddedType::BitShare =>  write_compressed_element_u8_u64(values),
        }
    }

    pub fn from_n(n: u16) -> Self {
        if n == 1 {
            EmbeddedType::BitShare
        } else if n < 256{
            EmbeddedType::U8
        } else {
            EmbeddedType::U16
        }
    }
}

/// Largest value stored in the table data.
fn max_of_ndarray(data: &NDArray<u16>) -> u16 {
    match data {
        NDArray::Leaf(v) => *v,
        NDArray::Node(items) => items.iter().map(max_of_ndarray).max().unwrap_or(0),
    }
}

/// Generation-time accounting for the exported `N_MAX`: the framework derives
/// the B2A conversion bit width from `N_MAX` without ever scanning the table
/// online, so `N_MAX` is defined as the largest value actually stored in the
/// table (the realized tail cutoff). This is generally SMALLER than the
/// considered range end `n` documented in the header comment (`n` is the
/// representation bound the SD was computed over, typically `2^b - 1`).
/// The hard invariant is that the data stays within the considered range;
/// a violation means the fill's breaking condition and the written values
/// disagree — abort the export rather than ship a table whose metadata lies.
fn checked_table_max(lut: &LookupTable<u16>, n: u16) -> u16 {
    let table_max = max_of_ndarray(&lut.data);
    assert!(
        table_max <= n,
        "largest table value {} exceeds the considered range end n = {}",
        table_max, n
    );
    table_max
}

fn to_camel_case(s: &str) -> String {
    s.split(|c: char| !c.is_ascii_alphanumeric()) // split on '_' '-' ' ' etc.
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
            }
        })
        .collect()
}



// takes a vector of u16 values, expects it to represent u8 values 
// and writes each 8 value chunk as a u64
fn write_compressed_element_u8_u64(values: &Vec<u16>) -> String {
    values.chunks(8).map(|chunk| {
        // Zero-pad a final partial chunk (last dimension smaller than RATIO).
        let mut bytes = [0u8; 8];
        for (i, c) in chunk.iter().enumerate() {
            bytes[i] = (*c % 256) as u8;
        }
        format!("{:#018x}", u64::from_le_bytes(bytes))
    }).collect::<Vec<_>>().join(", ")
}

// takes a vector of u16 values,
// and writes each 4 value chunk as a u64
fn write_compressed_element_u16_u64(values: &Vec<u16>) -> String {
    values.chunks(4).map(|chunk| {
        let mut bytes =  [0u8; 8];
        for (i, c) in chunk.iter().enumerate(){
            bytes[2*i] = *c as u8;
            bytes[2*i+1] = (*c >> 8) as u8;
        }
        format!("{:#018x}", u64::from_le_bytes(bytes))
    }).collect::<Vec<_>>().join(", ")
}


// Odd alignment to preserve string in created file. 
fn write_preamble<const S: usize>(
    file: &mut BufWriter<File>, 
    embedded_type: EmbeddedType, 
    k: &Vec<usize>, 
    skew: u32,
    l: usize,
    cube: bool,
    delta: &Decimal<S>,
    n: u16,
    table_max: u16
) -> std::io::Result<()> {
let embedded = embedded_type.to_str();
let d = if cube { 3 } else { 2 };
writeln!(
file,
"// SD(Z, Pi_Z) = {} < 2^{} considered range: [0,{}]", 
delta.to_scientific_notation(), 
neg_power_two(&delta),
n
)?;
writeln!(
file,
"use crate::lut_sampler::tables;
use rss_lut::share::gf_template::GFT;
type Wrapper = u64;
type Embedded = {};
const RATIO: usize = std::mem::size_of::<Wrapper>() / std::mem::size_of::<Embedded>();
pub type GF = GFT<Wrapper, Embedded, RATIO>;
pub const K: [usize; {}] = {:?};
pub const SKEW: usize = {};
pub const L: usize = {};
pub const D: usize = {};
pub const N_MAX: u16 = {};\n",
embedded, k.len(), k, skew, l, d, table_max
)
}

pub fn write_matrix_lut_to_rust_file<const S: usize>(
    lut: LookupTable<u16>, 
    path: &Option<String>, 
    embedded_type: EmbeddedType, 
    skew: u32, 
    l: usize,
    delta: &Decimal<S>,
    n: u16
) -> std::io::Result<()> {
    assert!(lut.k.len() == 2, "must have two k values for matrix export");
    let table_max = checked_table_max(&lut, n);
    let path = match path {
        Some(p) => p,
        None => panic!("No path provided for writing LUT to file"),
    };
    let mut path_buffer = PathBuf::from(path);
    path_buffer.set_extension("rs");
    let struct_name = Path::new(&path)
        .file_name()
        .and_then(|s| s.to_str())
        .map(to_camel_case)
        .unwrap_or_else(|| panic!("Invalid path/filename for struct name: {path}"));
    let mut file = BufWriter::new(File::create(path_buffer)?);
    let size_1 = 1 << lut.k[0];
    let size_2 = 1usize << lut.k[1];
    let ratio: usize = embedded_type.ratio();
    let size2_red = size_2.div_ceil(ratio);
    write_preamble(&mut file, embedded_type, &lut.k, skew, l, false, delta, n, table_max)?;
writeln!(
file,
"pub const SIZE1: usize = {};
pub const SIZE2: usize = {};
pub const SIZE2_RED: usize = {};\n
pub struct {};\n
impl tables::Matrix<SIZE1, SIZE2, SIZE2_RED> for {}{{
\tconst SIZE1: usize = SIZE1;
\tconst SIZE2: usize = SIZE2;
\tconst SIZE2_RED: usize = SIZE2_RED;
\tconst RATIO: usize = RATIO;
\ttype GF = GF;
\tconst K: [usize; 2] = K;
\tconst SKEW: usize = SKEW;
\tconst L: usize = L;
\tconst D: usize = D;
\tconst N_MAX: u16 = N_MAX;
",
size_1, size_2, size2_red, struct_name, struct_name
)?;
    writeln!(
        file,
        "\tconst LUT_TABLE: [[u64; {}]; {}] = [",
        size2_red, size_1
    )?;

    let matrix = match lut.data {
        NDArray::Leaf(_) => panic!("Error, expected 2D LUT"),
        NDArray::Node(items) => items,
    };

    for row in 0..size_1{
        let vector = match &matrix[row] {
            NDArray::Leaf(_) => panic!("Error, expected 2D LUT"),
            NDArray::Node(items) => items,
        };
        let mut values = Vec::with_capacity(size_2);

        for col in 0..size_2{
            let value = match &vector[col] {
                NDArray::Leaf(val) => *val,
                NDArray::Node(_) => panic!("Error, expected 2D LUT"),
            };
            values.push(value);
        }
        let formatted = embedded_type.write_compressed_element_to_u64(&values);
        writeln!(file, "\t/*row = {}*/           [{}],", row, formatted)?;
    }
    writeln!(file, "\t];\n}}")?;
    Ok(())
}

pub fn write_cube_lut_to_rust_file<const S: usize>(
    lut: LookupTable<u16>, 
    path: &Option<String>, 
    embedded_type: EmbeddedType, 
    skew: u32, 
    l: usize,
    delta: &Decimal<S>,
    n: u16
) -> std::io::Result<()> {
    assert!(lut.k.len() == 3, "must have 3 k values to export cube LUT");
    let table_max = checked_table_max(&lut, n);
    let path = match path {
        Some(p) => p,
        None => panic!("No path provided for writing LUT to file"),
    };
    let struct_name = Path::new(&path)
        .file_name()
        .and_then(|s| s.to_str())
        .map(to_camel_case)
        .unwrap_or_else(|| panic!("Invalid path/filename for struct name: {path}"));
    let mut path_buffer = PathBuf::from(path);
    path_buffer.set_extension("rs");
    let mut file = BufWriter::new(File::create(path_buffer)?);
    let k = lut.k;
    let size_1 = 1 << k[0];
    let size_2 = 1 << k[1];
    let size_3 = 1usize << k[2];
    let ratio: usize = embedded_type.ratio();
    let size3_red = size_3.div_ceil(ratio);
    write_preamble(&mut file, embedded_type, &k, skew, l, true, delta, n, table_max)?;
writeln!(
file,
"pub const SIZE1: usize = {};
pub const SIZE2: usize = {};
pub const SIZE3: usize = {};
pub const SIZE3_RED: usize = {};\n
pub struct {};\n
impl tables::Cube<SIZE1, SIZE2, SIZE3, SIZE3_RED> for {}{{
\tconst SIZE1: usize = SIZE1;
\tconst SIZE2: usize = SIZE2;
\tconst SIZE3: usize = SIZE3;
\tconst SIZE3_RED: usize = SIZE3_RED;
\tconst RATIO: usize = RATIO;
\ttype GF = GF;
\tconst K: [usize; 3] = K;
\tconst SKEW: usize = SKEW;
\tconst L: usize = L;
\tconst D: usize = D;
\tconst N_MAX: u16 = N_MAX;",
size_1, size_2, size_3, size3_red, struct_name, struct_name
)?;
    writeln!(
        file,
        "\tconst LUT_TABLE: [[[u64; {}]; {}]; {}] = [",
        size3_red, size_2, size_1
    )?;

    let cube = match lut.data {
        NDArray::Leaf(_) => panic!("Error, expected 3D LUT"),
        NDArray::Node(items) => items,
    };

    for row in 0..size_1{
        let matrix = match &cube[row] {
            NDArray::Leaf(_) => panic!("Error, expected 3D LUT"),
            NDArray::Node(items) => items,
        };
        writeln!(file, "\t// row = {}\n\t\t[", row)?;
        for col in 0..size_2{
            let vector = match &matrix[col] {
                NDArray::Leaf(_) => panic!("Error, expected 3D LUT"),
                NDArray::Node(items) => items,
            };
            let mut values = Vec::with_capacity(size_3);
            for lay in 0..size_3{
                let value = match &vector[lay] {
                    NDArray::Leaf(val) => *val,
                    NDArray::Node(_) => panic!("Error, expected 3D LUT"),
                };
                values.push(value);
            }
            let formatted = embedded_type.write_compressed_element_to_u64(&values);
            writeln!(file, "\t\t/*col = {}*/           [{}],", col, formatted)?;
        }
        writeln!(file, "\t\t],")?;

    }
    writeln!(file, "\t];\n}}")?;
    Ok(())
}
