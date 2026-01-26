use std::sync::Mutex;
use std::time::Instant;
use indexmap::IndexMap;
use once_cell::sync::Lazy;

struct Profiler {
    instants: IndexMap<String, Instant>,
    times: IndexMap<String, f64>,
}

pub static PROFILER: Lazy<Mutex<Profiler>> = Lazy::new(|| Mutex::new(Profiler::new()));

impl Profiler {
    fn new() -> Self {
        Self { instants: IndexMap::new(), times: IndexMap::new() }
    }

    fn start(&mut self, name: &str) {
        self.instants.insert(name.to_string(), Instant::now());
    }

    fn stop(&mut self, name: &str) {
        let elapsed = self.instants.get(name).unwrap().elapsed();
        self.times.insert(name.to_string(), elapsed.as_secs_f64());
    }
}

pub fn get_delta_time(name: &str) -> f64 {
    match PROFILER.lock().unwrap().times.get(name) {
        Some(time) => *time,
        None => 0.0,
    }
}

pub fn profiler_start(name: &str) {
    PROFILER.lock().unwrap().start(name);
}

pub fn profiler_stop(name: &str) {
    PROFILER.lock().unwrap().stop(name);
}

pub fn print_profile() {
    println!("PROFILER:");
    for (name, time) in PROFILER.lock().unwrap().times.iter() {
        println!("  {}: {:.0}ms", name, time * 1000.0);
    }
}
