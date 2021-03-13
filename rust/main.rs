extern crate itertools;
extern crate rand;

use itertools::Itertools;
use std::time::SystemTime;
use rand::{Rng, thread_rng};

fn main() {
    let start_time = SystemTime::now();
    let lengths: [[i64; 2]; 1] = [[3, 4]];
    let iterations: i64 = 1000;
    for [n, k] in lengths.iter() {
        let mut combs: Vec<_> = (0..*k).combinations(1).collect_vec();
        for bsl_length in 2..(k + 2) {
            combs.extend((0..*k).combinations(bsl_length as usize).collect_vec())
        }
        let mut total_ls: i64 = 0;
        for iteration_number in 0..iterations {
            let current_iteration_time = SystemTime::now();
            let mut rng = thread_rng();
            let fs: i64 = rng.gen_range(0, 1 << (2 * n * k));
            let mut master_pair = MasterPair {key_length: *k + 1, file_length: 2 * k,
                file: fs % (1 << (2 * k)), batch_size: *n, saved_bsl: vec![], saved_key: 0,
                bsl_length_sum: i64::MAX};
            let mut children = Vec::new();
            for a in 1..*n {
                let key_child = Key {value: 0, length: *k,
                    file: (fs >> (2 * k * a)) % (1 << (2 * k)), calculated: 0,};
                let bsl_child: BSL = BSL {bsl_iterable: (*combs).to_vec(), saved_value: vec![],
                    child: key_child, current_bsl: vec![]};
                children.push(bsl_child);
            }

            'bsl_loop: for current_bsl in combs.iter() {
                if current_bsl.len() as i64 + master_pair.batch_size - 1 < master_pair.bsl_length_sum {
                    for current_key in 0..(1 << master_pair.key_length) {
                        let mut current_bsl_len: i64 = current_bsl.len() as i64;
                        let f0c: i64 = (master_pair.file ^ xor_sum(current_key, current_bsl.to_vec())) as i64;
                        let mut i: i64 = 0;
                        let mut res: IterRes = IterRes {passed: false, bsl_length: 0};
                        'child_loop: for child_index in 0..children.len() as i64 {
                            res = children[child_index as usize].iterate(f0c, current_bsl_len, master_pair.batch_size - i - 2, master_pair.bsl_length_sum);
                            i += 1;
                            if !res.passed {
                                break 'child_loop;
                            }
                            else {
                                current_bsl_len += res.bsl_length;
                            }
                        }
                        if res.passed {
                            master_pair.bsl_length_sum = current_bsl_len;
                            master_pair.saved_bsl = current_bsl.to_vec();
                            master_pair.saved_key = current_key;
                            for child_index in 0..children.len() as i64 {
                                children[child_index as usize].save();
                            }

                            if master_pair.bsl_length_sum == master_pair.batch_size {
                                break 'bsl_loop;
                            }
                        }
                    }
                }
            }

            println!("");

            total_ls += master_pair.saved_bsl.len() as i64;
            for child_index in 0..children.len() as i64 {
                total_ls += children[child_index as usize].saved_value.len() as i64;
            }

            match current_iteration_time.elapsed() {
                Ok(elapsed) => {println!("{} -> {:?}ms", iteration_number + 1, elapsed.as_millis() / 1000);}
                Err(e) => {println!("Error: {} -> {:?}", iteration_number + 1, e);}
            }

            println!("{:?}\n{:?}\n{:?} || {:?} | {:?}", bit_rep(fs), bit_rep(master_pair.file ^ xor_sum(master_pair.saved_key, (*master_pair.saved_bsl).to_vec())), bit_rep8(master_pair.file), bit_rep(master_pair.saved_key), master_pair.saved_bsl);

            for child_index in 0..children.len() as i64 {
                println!("{:?} || {:?} | {:?}", bit_rep8(children[child_index as usize].child.file), bit_rep(children[child_index as usize].child.value), children[child_index as usize].saved_value);
            }
            println!("");

        }
        println!("Average L(s) for n({}) and l({}) -> {:?}", n, k, total_ls as f64 / iterations as f64);
    }

    match start_time.elapsed() {
        Ok(elapsed) => {println!("{:?}ms total. Average {}ms per iteration", elapsed.as_millis(), elapsed.as_millis() as f64 / 1000f64);}
        Err(e) => {println!("Error: {:?}", e);}
    }
}

fn xor_sum(key: i64, bit_shift_list: Vec<i64>) -> i64 {
    let mut out: i64 = 0;
    for bit_shift in bit_shift_list.iter() {
        out ^= key << bit_shift;
    }
    return out;
}

fn bit_rep(item: i64) -> String {
    return format!("{:#07b}", item);
}

fn bit_rep8(item: i64) -> String {
    return format!("{:#010b}", item);
}

struct MasterPair {
    key_length: i64,
    file_length: i64,
    file: i64,
    batch_size: i64,
    saved_bsl: Vec<i64>,
    saved_key: i64,
    bsl_length_sum: i64,
}

struct BSL {
    bsl_iterable: Vec<Vec<i64>>,
    saved_value: Vec<i64>,
    child: Key,
    current_bsl: Vec<i64>
}
impl BSL {
    fn iterate(&mut self, f0c: i64, previous_bsl_length: i64, index_bsl_threshold: i64, current_best: i64) -> IterRes {
        let dn: i64 = f0c ^ self.child.file;
        let min_s: i64 = dn.trailing_zeros() as i64;
        let max_s: i64 = 64 - dn.leading_zeros() as i64 - self.child.length - 1;
        for current_bsl in self.bsl_iterable.iter() {
            self.current_bsl = current_bsl.to_vec();
            if index_bsl_threshold + current_bsl.len() as i64 + previous_bsl_length < current_best {
                if current_bsl.first().copied() <= Some(min_s) && current_bsl.last().copied() >= Some(max_s) {
                    let res: bool = self.child.iterate(dn, current_bsl.to_vec());
                    if res {
                        return IterRes {passed: true, bsl_length: current_bsl.len() as i64};
                    }
                }
            }
        }
        return IterRes {passed: false, bsl_length: 0};
    }

    fn save(&mut self) -> () {
        self.saved_value = self.current_bsl.to_vec();
        self.child.save();
    }
}

struct Key {
    value: i64,
    length: i64,
    file: i64,
    calculated: i64,
}
impl Key {
    fn iterate(&mut self, mut dn: i64, mut related_bsl: Vec<i64>) -> bool {
        if related_bsl.len() as i64 == 1 {
            self.calculated = dn >> related_bsl[0];
            return true;
        }
        else {
            self.calculated = 0;
            let sub: i64 = related_bsl[0];
            dn >>= sub;
            for index in 0..related_bsl.len() as i64 {
                related_bsl[index as usize] -= sub;
            }
            for index in 0..self.length + related_bsl[(related_bsl.len() as i64 - 1) as usize] {
                self.calculated += (((dn ^ self.xor_sum_limited(self.calculated, (*related_bsl).to_vec(), index)) >> index) % 2) << index;
            }
            for index in 0..related_bsl.len() as i64 {
                related_bsl[index as usize] += sub;
            }
            if xor_sum(self.calculated, related_bsl) != dn {
                return false;
            }
            return true;
        }
    }

    fn xor_sum_limited(&self, key: i64, bsl: Vec<i64>, threshold: i64) -> i64 {
        let mut out: i64 = 0;
        for bit_shift in bsl.iter() {
            if bit_shift > &threshold {
                break;
            }
            else if bit_shift + self.length > threshold {
                out ^= key << bit_shift;
            }
        }
        return out;
    }

    fn save(&mut self) -> () {
        self.value = self.calculated;
    }
}

struct IterRes {
    passed: bool,
    bsl_length: i64
}
