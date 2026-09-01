//! CLUX Intermediate Representation & Zero-Allocation Memory Arena

use std::cell::UnsafeCell;

/// Static Pre-Allocated Arena for Bare-Metal execution without OS malloc overhead
pub struct StaticArena {
    memory: UnsafeCell<Vec<u8>>,
    offset: std::cell::Cell<usize>,
}

impl StaticArena {
    pub fn new(capacity: usize) -> Self {
        let mut mem = Vec::with_capacity(capacity);
        unsafe { mem.set_len(capacity); }
        Self {
            memory: UnsafeCell::new(mem),
            offset: std::cell::Cell::new(0),
        }
    }

    pub fn alloc_f32_slice(&self, len: usize) -> &mut [f32] {
        let bytes_needed = len * std::mem::size_of::<f32>();
        let current = self.offset.get();
        
        // Align to 32-byte boundary for AVX2 SIMD compatibility
        let align_offset = (32 - (current % 32)) % 32;
        let start = current + align_offset;
        let end = start + bytes_needed;
        
        self.offset.set(end);

        unsafe {
            let ptr = (*self.memory.get()).as_mut_ptr().add(start) as *mut f32;
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    pub fn reset(&self) {
        self.offset.set(0);
    }
}
