use std::collections::HashMap;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::time::{Instant};

fn is_whitespace(c: char) -> bool {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

fn main() -> std::io::Result<()> {
    let file_path = "gutenberg-500M.txt";
    let file = File::open(file_path)?;
    //let file_size = fs::metadata(file_path)?.len();
    let buffer_size: usize = 32 * 1024 * 1024;

    let rdstart = Instant::now();

    let mut reader = BufReader::with_capacity(buffer_size, file);
    let mut word_counts: HashMap<String, u64> = HashMap::new();
    {
        let mut current_word = String::with_capacity(64);
        let mut buffer= [0 as u8; 1024];
        loop { match reader.read(&mut buffer) {
            Ok(bytes_read) => {
                if bytes_read == 0 {
                    break;
                }
                for i in 0..bytes_read
                {
                    let c = buffer[i];
                    if is_whitespace(c as char) {
                        if !current_word.is_empty() {
                            let v = word_counts.entry(current_word).or_insert(0);
                            *v += 1;
                            current_word = String::with_capacity(64)
                        }
                        else {
                            current_word.clear()
                        }
                    }
                    else {
                        current_word.push(c as char);
                    }
                }
            }
            Err(err) => {
                return Err(err);
            }
        }}
        if !current_word.is_empty() {
            let v = word_counts.entry(current_word).or_insert(0);
            *v += 1;
        }
    }

    let rdend = Instant::now();

    let sstart = Instant::now();
    let mut count_vec: Vec<(&String, &u64)> = word_counts.iter().collect();
    count_vec.sort_by(|a, b| b.1.cmp(a.1));
    let send = Instant::now();

    println!("Read + distribute: {}s", (rdend - rdstart).as_secs_f64());
    println!("Sort: {}s", (sstart - send).as_secs_f64());
    for i in 0..10 {
        println!("{}: {}", count_vec[i].0, count_vec[i].1);
    }

    Ok(())
}