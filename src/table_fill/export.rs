use core::panic;
use std::fs::File;
use std::io::{BufWriter, Write};

use super::{LookupTable, NDArray};

pub fn print_dimensions(lut: &LookupTable<u8>) {
    let size = 1 << lut.k;

    assert!(size > 10);
    // 5 if lay is set
    println!("[254,255,8]: {}",lut.get(&[254,255,8]));

    // 13 if col is set
    println!("[255,111,255]: {}",lut.get(&[255,111,255]));

    // 10 if lay is set
    println!("[226,255,255]: {}",lut.get(&[226,255,255]));
}

pub fn write_lut_to_rust_file(lut: LookupTable<u8>, path: &str) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    let size = 1 << lut.k;
    let red_size = size / 8;
    writeln!(
        file,
        "pub const LUT_TABLE: [[[u64; {}]; {}]; {}] = [",
        red_size, size, size
    )?;

    let cube = match lut.data {
        NDArray::Leaf(_) => panic!("Error, expected 3D LUT"),
        NDArray::Node(items) => items,
    };

    for row in 0..size{
        let matrix = match &cube[row] {
            NDArray::Leaf(_) => panic!("Error, expected 3D LUT"),
            NDArray::Node(items) => items,
        };
        writeln!(file, "// row = {}\n   [", row)?;
        for col in 0..size{
            let vector = match &matrix[col] {
                NDArray::Leaf(_) => panic!("Error, expected 3D LUT"),
                NDArray::Node(items) => items,
            };
            let mut values = Vec::with_capacity(size);
            for lay in 0..size{
                let value = match &vector[lay] {
                    NDArray::Leaf(val) => *val,
                    NDArray::Node(_) => panic!("Error, expected 3D LUT"),
                };
                values.push(value);
            }
            let formatted = values.chunks(8).map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("Expected chunk of 8 bytes");
                format!("{:#018x}", u64::from_le_bytes(bytes))
            }).collect::<Vec<_>>().join(", ");
            writeln!(file, " /*col = {}*/           [{}],", col, formatted)?;
        }
        writeln!(file, "    ],")?;

    }
    writeln!(file, "];")?;
    Ok(())
}
