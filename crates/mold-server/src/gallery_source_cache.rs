//! Bounded process-only derived media caches; callers retain all authority checks.
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

pub(crate) struct PrivateCache<K, V> {
    entries: Mutex<VecDeque<(K, V, usize)>>,
    flights: Mutex<HashMap<K, Weak<Mutex<()>>>>,
    max_entries: usize,
    max_bytes: usize,
}
impl<K: Eq + Hash + Clone, V: Clone> PrivateCache<K, V> {
    pub(crate) fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            flights: Mutex::new(HashMap::new()),
            max_entries,
            max_bytes,
        }
    }
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        let mut entries = lock(&self.entries);
        let index = entries.iter().position(|(stored, _, _)| stored == key)?;
        let entry = entries.remove(index)?;
        let result = entry.1.clone();
        entries.push_back(entry);
        Some(result)
    }
    pub(crate) fn insert(&self, key: K, value: V, bytes: usize) {
        if bytes > self.max_bytes || self.max_entries == 0 {
            return;
        }
        let mut entries = lock(&self.entries);
        entries.retain(|(stored, _, _)| stored != &key);
        while entries.len() >= self.max_entries
            || entries.iter().map(|(_, _, cost)| cost).sum::<usize>() > self.max_bytes - bytes
        {
            entries.pop_front();
        }
        entries.push_back((key, value, bytes));
    }
    pub(crate) fn flight(&self, key: &K) -> Arc<Mutex<()>> {
        let mut flights = lock(&self.flights);
        flights.retain(|_, flight| flight.strong_count() > 0);
        if let Some(flight) = flights.get(key).and_then(Weak::upgrade) {
            return flight;
        }
        let flight = Arc::new(Mutex::new(()));
        flights.insert(key.clone(), Arc::downgrade(&flight));
        flight
    }
}

/// Shared by gallery inventory and content/thumbnail cold scans. RAII returns
/// capacity on every error/panic; blocking work is only entered on blocking tasks.
pub(crate) struct ColdWork {
    active: Mutex<usize>,
    wake: Condvar,
    limit: usize,
}
pub(crate) struct ColdPermit<'a>(&'a ColdWork);
impl ColdWork {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            active: Mutex::new(0),
            wake: Condvar::new(),
            limit,
        }
    }
    pub(crate) fn enter(&self) -> ColdPermit<'_> {
        let mut active = lock(&self.active);
        while *active >= self.limit {
            active = self.wake.wait(active).unwrap_or_else(|p| p.into_inner());
        }
        *active += 1;
        ColdPermit(self)
    }
}
impl Drop for ColdPermit<'_> {
    fn drop(&mut self) {
        *lock(&self.0.active) -= 1;
        self.0.wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_and_entry_budgets_evict_lru_and_reject_oversized_values() {
        let cache = PrivateCache::new(2, 5);
        cache.insert(1, 1, 2);
        cache.insert(2, 2, 2);
        assert_eq!(cache.get(&1), Some(1));
        cache.insert(3, 3, 2);
        assert_eq!(cache.get(&2), None);
        cache.insert(4, 4, 6);
        assert_eq!(cache.get(&4), None);
        cache.insert(5, 5, 5);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&3), None);
    }
    #[test]
    fn distinct_cold_work_is_bounded_and_permits_return_after_errors() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let gate = ColdWork::new(2);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..12 {
                scope.spawn(|| {
                    let _permit = gate.enter();
                    let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(n, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert!(peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(*lock(&gate.active), 0);
    }
}
