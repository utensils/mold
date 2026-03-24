use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::hash::{DefaultHasher, Hasher};

pub(crate) const DEFAULT_PROMPT_CACHE_CAPACITY: usize = 16;
pub(crate) const DEFAULT_IMAGE_CACHE_CAPACITY: usize = 8;

pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes);
    hasher.finish()
}

#[derive(Debug)]
pub(crate) struct LruCache<K, V> {
    capacity: usize,
    order: VecDeque<K>,
    entries: HashMap<K, V>,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    pub(crate) fn get_cloned(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let value = self.entries.get(key).cloned()?;
        self.touch(key);
        Some(value)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) {
        if self.entries.insert(key.clone(), value).is_some() {
            self.order.retain(|existing| existing != &key);
        }
        self.order.push_back(key);
        self.evict_if_needed();
    }

    pub(crate) fn clear(&mut self) {
        self.order.clear();
        self.entries.clear();
    }

    fn touch(&mut self, key: &K) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct CachedTensor {
    tensor: Tensor,
}

impl CachedTensor {
    pub(crate) fn from_tensor(tensor: &Tensor) -> Result<Self> {
        Ok(Self {
            tensor: tensor.to_device(&Device::Cpu)?,
        })
    }

    pub(crate) fn restore(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        Ok(self.tensor.to_device(device)?.to_dtype(dtype)?)
    }
}

#[derive(Clone)]
pub(crate) struct CachedTensorPair {
    pub(crate) first: CachedTensor,
    pub(crate) second: CachedTensor,
}

impl CachedTensorPair {
    pub(crate) fn from_tensors(first: &Tensor, second: &Tensor) -> Result<Self> {
        Ok(Self {
            first: CachedTensor::from_tensor(first)?,
            second: CachedTensor::from_tensor(second)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_cache_evicts_oldest_entry() {
        let mut cache = LruCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        assert!(cache.get_cloned(&"a").is_none());
        assert_eq!(cache.get_cloned(&"b"), Some(2));
        assert_eq!(cache.get_cloned(&"c"), Some(3));
    }

    #[test]
    fn lru_cache_updates_recently_used_order() {
        let mut cache = LruCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.get_cloned(&"a"), Some(1));
        cache.insert("c", 3);

        assert_eq!(cache.get_cloned(&"a"), Some(1));
        assert!(cache.get_cloned(&"b").is_none());
        assert_eq!(cache.get_cloned(&"c"), Some(3));
    }
}
