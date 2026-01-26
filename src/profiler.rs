use std::sync::Mutex;
use std::time::Instant;
use indexmap::IndexMap;
use once_cell::sync::Lazy;

struct Profiler {
    stack: Vec<(String, Instant)>,
    frame_entries: IndexMap<(String, usize), (f64, u32)>,
    frame_order: Vec<(String, usize)>,
    persistent_entries: IndexMap<(String, usize), (f64, u32)>,
    persistent_order: Vec<(String, usize)>,
    delta_time: f64,
}

pub static PROFILER: Lazy<Mutex<Profiler>> = Lazy::new(|| Mutex::new(Profiler::new()));

impl Profiler {
    fn new() -> Self {
        Self { stack: Vec::new(), frame_entries: IndexMap::new(), frame_order: Vec::new(), persistent_entries: IndexMap::new(), persistent_order: Vec::new(), delta_time: 0.0}
    }

    fn start(&mut self, name: &str) {
        self.stack.push((name.to_string(), Instant::now()));
    }

    fn stop(&mut self) {
        let (name, start) = self.stack.pop().unwrap();
        let depth = self.stack.len();
        let elapsed = start.elapsed().as_secs_f64();

        if name == "main" {
            self.delta_time = elapsed;
        }

        let (entries, order) = if self.inside_main() || name == "main" {
            (&mut self.frame_entries, &mut self.frame_order)
        } else {
            (&mut self.persistent_entries, &mut self.persistent_order)
        };

        let key = (name.clone(), depth);

        match entries.get_mut(&key) {
            Some(entry) => {
                entry.0 += elapsed;
                entry.1 += 1;
            }
            None => {
                entries.insert(key.clone(), (elapsed, 1));
                order.push(key);
            }
        }
    }


    fn print(&self) {
        if !self.persistent_order.is_empty() {
            println!("PROFILER (persistent):");
            self.print_layer(&self.persistent_entries, &self.persistent_order);
        }

        println!("PROFILER (frame):");
        self.print_layer(&self.frame_entries, &self.frame_order);
    }

    fn print_layer(&self, entries: &IndexMap<(String, usize), (f64, u32)>, order: &Vec<(String, usize)>) {
        for (name, depth) in order {
            let (time, count) = entries[&(name.clone(), *depth)];
            let indent = "  ".repeat(*depth);

            print!("{}{}: {:.2}ms", indent, name, time * 1000.0);
            if count > 1 {
                print!(" x{}", count);
            }
            println!();
        }
    }

    fn inside_main(&self) -> bool {
        self.stack.iter().any(|(name, _)| name == "main")
    }
}

pub fn get_delta_time() -> f64 {
    PROFILER.lock().unwrap().delta_time
}

pub fn profiler_start(name: &str) {
    PROFILER.lock().unwrap().start(name);
}

pub fn profiler_stop(_name: &str) {
    PROFILER.lock().unwrap().stop();
}

pub fn profiler_reset() {
    let mut profiler = PROFILER.lock().unwrap();
    profiler.frame_entries.clear();
    profiler.frame_order.clear();
    profiler.stack.clear();
}


pub fn print_profile() {
    PROFILER.lock().unwrap().print();
}

