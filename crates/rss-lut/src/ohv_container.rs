use std::marker::PhantomData;

use maestro::rep3_core::{network::NetSerializable, party::{broadcast::BroadcastContext, error::MpcResult, MainParty}, share::{RssShare, RssShareVec}};

use crate::{mult_verification::TripleVector, online::open_rss_many, share::{gf_template::{GFTTrait, Share, ShareType}, gf2p64::{GF2p64, GF2p64Subfield}, helper_types::AllowedTypes}, util::mul_triple_vec::DotProdRecorder};



#[derive(Copy, Clone, Debug)]
pub struct RndOhvOutput<T: GFTTrait, IndexType: AllowedTypes, const SIZE: usize> {
    /// share i of one-hot vector
    pub si: OhvVec<T, IndexType, SIZE>,
    /// share i+1 of one-hot vector
    pub sii: OhvVec<T, IndexType, SIZE>,
    /// (2,3) sharing of the position of the 1 in the vector
    pub random_offset: RssShare<ShareType<IndexType>>,
    _marker: PhantomData<T>
}

#[derive(Clone, Copy, Debug)]
pub struct OhvVec<T: GFTTrait, IndexType: AllowedTypes, const SIZE: usize>{
    pub inner: [bool; SIZE],
    _marker: PhantomData<T>,
    _index_marker: PhantomData<IndexType>,
}

pub struct MatrixOhv<T: GFTTrait, IndexType: AllowedTypes, const SIZE1: usize, const SIZE2: usize>{
    pub row_ohv: RndOhvOutput<T, IndexType, SIZE1>,
    pub col_ohv: RndOhvOutput<T, IndexType, SIZE2>,
    pub _marker: PhantomData<T>,
}

pub struct CubeOhv<T: GFTTrait, IndexType: AllowedTypes, const SIZE1: usize, const SIZE2: usize, const SIZE3: usize>{
    pub row_ohv: RndOhvOutput<T, IndexType, SIZE1>,
    pub col_ohv: RndOhvOutput<T, IndexType, SIZE2>,
    pub lay_ohv: RndOhvOutput<T, IndexType, SIZE3>,
    pub _marker: PhantomData<T>,
}

impl<T: GFTTrait, IndexType: AllowedTypes, const SIZE1: usize, const SIZE2: usize> MatrixOhv<T, IndexType, SIZE1, SIZE2>{
        // collapses rows
    pub fn collapse_rows_local<const SIZE2_RED: usize>(&self, lut_table: &[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE2_RED]; SIZE1]) -> [[T::Wrapper; SIZE2_RED];2]{
        let ohv = &self.row_ohv;
        [ohv.si.collapse_rows_matrix(lut_table), ohv.sii.collapse_rows_matrix(lut_table)]
    }

        // Collapses columns
    pub fn collapse_cols<const SIZE2_RED: usize>(&self, matrices: &[[T::Wrapper; SIZE2_RED];2], rec: &mut TripleVector) -> T::Embedded {
        let new_len= std::mem::size_of::<T::Embedded>();
        let mut res = T::Embedded::default();
        let ohv = &self.col_ohv;
        // Only the first SIZE entries are real; the packed wrapper may carry zero padding.
        let size = ohv.si.inner.len();

        // Construct v_i: &[T::Wrapper], v_ii: &[T::Wrapper]
        let v_i = &matrices[0];
        let v_ii = &matrices[1];

        let val_i = <T::Wrapper as NetSerializable>::as_byte_vec(v_i.iter(), v_i.len());
        let val_ii = <T::Wrapper as NetSerializable>::as_byte_vec(v_ii.iter(), v_ii.len());
        debug_assert!(val_i.len() == v_i.len()*std::mem::size_of::<T::Wrapper>());
        debug_assert!(val_ii.len() == v_ii.len()*std::mem::size_of::<T::Wrapper>());

        for (i, (vi, vii)) in val_i.chunks(new_len).zip(val_ii.chunks(new_len)).take(size).enumerate(){
            if ohv.si.inner[i]{
                res += T::Embedded::pack_bytes(vi);
                res += T::Embedded::pack_bytes(vii);
            } if ohv.sii.inner[i]{
                res += T::Embedded::pack_bytes(vi);
            }
        }
        match rec{
            TripleVector::MAL(r) =>{
                let flat_i: Vec<GF2p64>  = <T::Embedded as NetSerializable>::from_byte_vec(val_i, size).into_iter().take(size).map(|v| v.embed()).collect();
                let flat_ii: Vec<GF2p64> = <T::Embedded as NetSerializable>::from_byte_vec(val_ii, size).into_iter().take(size).map(|v| v.embed()).collect();
                let bi:      Vec<GF2p64> = ohv.si.inner.iter().map(|e| GF2p64::new(*e)).collect();
                let bii:     Vec<GF2p64> = ohv.sii.inner.iter().map(|e| GF2p64::new(*e)).collect();
                r.record_dot_in(
                    std::slice::from_ref(&bi), 
                    std::slice::from_ref(&bii), 
                    std::slice::from_ref(&flat_i), 
                    std::slice::from_ref(&flat_ii)
                );
            },
            TripleVector::SEMI(_) => {}
        }
        res
    }

    pub fn get_coordinates(&self, party: &mut MainParty, context: &mut BroadcastContext) -> MpcResult<Vec<ShareType<IndexType>>> {
        let row = self.row_ohv.random_offset;
        let col = self.col_ohv.random_offset;
        let v = vec![row, col];
        open_rss_many::<ShareType<IndexType>>(party, context, &v)
    }

    pub fn compute_offset(&mut self, offsets: &[RssShare<ShareType<IndexType>>]) -> RssShareVec<ShareType<IndexType>> {
        debug_assert_eq!(offsets.len(), 2);
        [&self.row_ohv.random_offset, &self.col_ohv.random_offset]
            .into_iter()
            .zip(offsets)
            .map(|(ohv, share)| RssShare {
                si: share.si + ohv.si,
                sii: share.sii + ohv.sii,
            })
            .collect()
    }

    pub fn rotate(&mut self, offset: &[ShareType<IndexType>]){
        self.row_ohv.rotate(offset[0]);
        self.col_ohv.rotate(offset[1]);
    }
    
    pub fn print(&self){
        println!("Row");
        self.row_ohv.print();
        println!("Col");
        self.col_ohv.print();
    }
}

impl<T: GFTTrait, IndexType: AllowedTypes, const SIZE1: usize, const SIZE2: usize, const SIZE3: usize> CubeOhv<T, IndexType, SIZE1, SIZE2, SIZE3>{
    // collapses columns
    pub fn collapse_columns_local<const SIZE3_RED: usize>(&self, lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2]]) -> [[[T::Wrapper; SIZE3_RED]; SIZE2];2]{
        let ohv = &self.col_ohv;
        [ohv.si.collapse_columns_cube(lut_table), ohv.sii.collapse_columns_cube(lut_table)]
    }

    // Collapses Rows
    pub fn collapse_rows<const SIZE3_RED: usize>(&self, matrices: &[[[T::Wrapper; SIZE3_RED]; SIZE2]; 2], rec: &mut TripleVector) -> [T::Wrapper; SIZE3_RED]{
        let mut res_vec = [T::Wrapper::default(); SIZE3_RED];
        let ohv = &self.row_ohv;
        match rec{
            TripleVector::SEMI(_) => {

                for(i, (layer_i, layer_ii)) in matrices[0].iter().zip(matrices[1]).enumerate(){
                    for (j, r) in res_vec.iter_mut().enumerate(){
                        if ohv.si.inner[i]{
                            *r += layer_i[j];
                            *r += layer_ii[j];
                        } if ohv.sii.inner[i] {
                            *r += layer_i[j];
                        }
                    }
                }
            },
            TripleVector::MAL(recorder) => {
                // The boolean values
                let mut ai  = vec![Vec::with_capacity(SIZE2); SIZE3_RED];
                let mut aii = vec![Vec::with_capacity(SIZE2); SIZE3_RED];
                // The LUT va
                let mut bi  = vec![Vec::with_capacity(SIZE2); SIZE3_RED];
                let mut bii = vec![Vec::with_capacity(SIZE2); SIZE3_RED];
                
                // i iterates 0..SIZE
                for(i, (layer_i, layer_ii)) in matrices[0].iter().zip(matrices[1]).enumerate(){
                    let ohvi = ohv.si.inner[i];
                    let ohvii = ohv.sii.inner[i];
                    let bool_si =  GF2p64::new(ohvi);
                    let bool_sii = GF2p64::new(ohvii);
                    for (inner_ai, inner_aii) in ai.iter_mut().zip(&mut aii){
                        inner_ai.push(bool_si);
                        inner_aii.push(bool_sii);
                    }
                    
                    // j iterates 0..SIZE_RED
                    for (j, r) in res_vec.iter_mut().enumerate(){                        
                        // let offset = j * T::RATIO;
                        // let embedded_layer_i = T::embed(&layer_i[j]); 
                        // let embedded_layer_ii = T::embed(&layer_ii[j]);
                        let embedded_layer_i = layer_i[j].embed(); 
                        let embedded_layer_ii = layer_ii[j].embed();
                        // debug_assert!(embedded_layer_i.len() == embedded_layer_ii.len());
                        // debug_assert!(embedded_layer_i.len() == T::RATIO);
                        bi[j].push(embedded_layer_i);
                        bii[j].push(embedded_layer_ii);
                        // for (k, (e_i, e_ii)) in embedded_layer_i.into_iter().zip(embedded_layer_ii).enumerate() {
                        //     bi[offset + k].push(e_i);
                        //     bii[offset + k].push(e_ii);
                        // }
                        
                        if ohvi{
                            *r += layer_i[j];
                            *r += layer_ii[j];
                        } 
                        if ohvii {
                            *r += layer_i[j];
                        }
                    }
                }
                assert!(bi.iter().all(|b| b.len() == SIZE2), "All bi vectors must be equal SIZE2: {}", SIZE2);
                // println!("lengths: ai {}, aii {}, bi {}, bii {}", ai[0].len(), aii[0].len(), bi[0].len(), bii[0].len());
                recorder.record_dot_in(&ai, &aii, &bi, &bii);
            }
        }
        
        res_vec
    }

    // #[instrument(name = "Collapse Columns", skip_all)]
    // Collapses Layers
    pub fn collapse_layers(&self, v_i: &[T::Wrapper], v_ii: &[T::Wrapper], rec: &mut TripleVector) -> T::Embedded{
        let new_len= std::mem::size_of::<T::Embedded>();
        let mut res = T::Embedded::default();
        let ohv = &self.lay_ohv;
        // Only the first SIZE entries are real; the packed wrapper may carry zero padding.
        let size = ohv.si.inner.len();
        let val_i = <T::Wrapper as NetSerializable>::as_byte_vec(v_i.iter(), v_i.len());
        let val_ii = <T::Wrapper as NetSerializable>::as_byte_vec(v_ii.iter(), v_ii.len());
        debug_assert!(val_i.len() == v_i.len()*std::mem::size_of::<T::Wrapper>());
        debug_assert!(val_ii.len() == v_ii.len()*std::mem::size_of::<T::Wrapper>());

        for (i, (vi, vii)) in val_i.chunks(new_len).zip(val_ii.chunks(new_len)).take(size).enumerate(){
            if ohv.si.inner[i]{
                res += T::Embedded::pack_bytes(vi);
                res += T::Embedded::pack_bytes(vii);
            } if ohv.sii.inner[i]{
                res += T::Embedded::pack_bytes(vi);
            }
        }
        match rec{
            TripleVector::MAL(r) =>{
                let flat_i: Vec<GF2p64>  = <T::Embedded as NetSerializable>::from_byte_vec(val_i, size).into_iter().take(size).map(|v| v.embed()).collect();
                let flat_ii: Vec<GF2p64>  = <T::Embedded as NetSerializable>::from_byte_vec(val_ii, size).into_iter().take(size).map(|v| v.embed()).collect();
                let bi:      Vec<GF2p64> = ohv.si.inner.iter().map(|e| GF2p64::new(*e)).collect();
                let bii:     Vec<GF2p64> = ohv.sii.inner.iter().map(|e| GF2p64::new(*e)).collect();
                // print_many(&flat_i, &flat_ii, &bi, &bii, &[T::Embedded::default().embed()], &[T::Embedded::default().embed()]);
                r.record_dot_in(
                    std::slice::from_ref(&bi), 
                    std::slice::from_ref(&bii), 
                    std::slice::from_ref(&flat_i), 
                    std::slice::from_ref(&flat_ii)
                );
            },
            TripleVector::SEMI(_) => {}
        }
        res
    }

    pub fn get_coordinates(&self, party: &mut MainParty, context: &mut BroadcastContext) -> MpcResult<Vec<ShareType<IndexType>>> {
        let row = self.row_ohv.random_offset;
        let col = self.col_ohv.random_offset;
        let lay = self.lay_ohv.random_offset;
        let v = vec![row, col, lay];
        open_rss_many::<ShareType<IndexType>>(party, context, &v)
    }

    pub fn compute_offset(&mut self, offsets: &[RssShare<ShareType<IndexType>>]) -> RssShareVec<ShareType<IndexType>> {
        debug_assert_eq!(offsets.len(), 3);
        [&self.row_ohv.random_offset, &self.col_ohv.random_offset, &self.lay_ohv.random_offset]
            .into_iter()
            .zip(offsets)
            .map(|(ohv, share)| {
                // ohv.print();
                // println!("skewed offset {} {}", share.si.to_u8(), share.sii.to_u8());
                RssShare {
                    si: share.si + ohv.si,
                    sii: share.sii + ohv.sii,
                }
            }).collect()
    }

    pub fn rotate(&mut self, offset: &[ShareType<IndexType>]){
        self.row_ohv.rotate(offset[0]);
        self.col_ohv.rotate(offset[1]);
        self.lay_ohv.rotate(offset[2]);
    }

    pub fn print(&self){
        println!("Row");
        self.row_ohv.print();
        println!("Col");
        self.col_ohv.print();
        println!("Lay");
        self.lay_ohv.print();
    }

}


impl<T:GFTTrait, IndexType: AllowedTypes, const SIZE: usize> RndOhvOutput<T, IndexType, SIZE>{
    pub fn new(ohv_output: (OhvVec<T, IndexType, SIZE>,OhvVec<T, IndexType, SIZE>), index: RssShare<ShareType<IndexType>>) -> Self { 
        Self{
            si: ohv_output.0,
            sii: ohv_output.1,
            random_offset: index,
            _marker: PhantomData,
        }
    }
    pub fn rotate(&mut self, offset: ShareType<IndexType>) {
        // println!("offset {:?}", offset);
        self.random_offset.si += offset;
        self.random_offset.sii += offset;
        debug_assert!(offset.inner().to_usize() < SIZE, "Offset in rotation is larger than truncation allows {} >= {}", offset.inner().to_usize(), SIZE);
        self.si.rotate(offset);
        self.sii.rotate(offset);
    }
    pub fn print(&self){
        self.si.print();
        self.sii.print();
        println!("Random offset: {:?}", self.random_offset);
    }
}

impl<T: GFTTrait, IndexType: AllowedTypes, const SIZE: usize> OhvVec<T,IndexType,SIZE>
where T::Embedded: Share, T::Wrapper: Share {
    pub fn new(val: [bool; SIZE]) -> Self{
        Self { inner: val, _marker: PhantomData, _index_marker: PhantomData }
    }

    pub fn rotate(&mut self, offset: ShareType<IndexType>){
        let mut tmp = [false; SIZE];
        for (i, el) in self.inner.iter().enumerate(){
            let index = i ^ offset.inner().to_usize();
            tmp[index] = *el;
        }
        self.inner = tmp;
    }

    pub fn collapse_columns_cube<const SIZE_RED: usize>(
        &self, 
        lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]]
    ) -> [[T::Wrapper; SIZE_RED]; SIZE]
    {
        let mut res_matrix = [[T::Wrapper::default(); SIZE_RED]; SIZE];
        for (layer, res) in lut_table.iter().zip(res_matrix.iter_mut()){
            for (i, row) in layer.iter().enumerate(){
                if self.inner[i] {
                    res.iter_mut().zip(row.iter()).for_each(|(r, &val)| *r += T::Wrapper::new(val));
                }
            }
        }
        // println!("{:?}", res_matrix);
        res_matrix
    }

    pub fn collapse_rows_matrix<const SIZE_RED: usize>(
        &self, 
        lut_table: &[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]
    ) -> [T::Wrapper; SIZE_RED]
    {
        let mut res_vec = [T::Wrapper::default(); SIZE_RED];
        for (i, row) in lut_table.iter().enumerate(){
            if self.inner[i] {
                res_vec.iter_mut().zip(row.iter()).for_each(|(r, &val)| *r += T::Wrapper::new(val));
            }
        }
        // println!("{:?}", res_matrix);
        res_vec
    }

    pub fn print(&self){
        print!("[");
        for i in self.inner.iter(){
            print!("{} ", *i as u8);
        }
        println!("]");
    }

}