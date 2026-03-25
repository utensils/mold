use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::hash::{DefaultHasher, Hasher};
use std::sync::Mutex;

pub(crate) const DEFAULT_PROMPT_CACHE_CAPACITY: usize = 16;
pub(crate) const DEFAULT_IMAGE_CACHE_CAPACITY: usize = 8;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct PromptCacheKey {
    prompt: String,
    guidance_bits: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct ImageSizeCacheKey {
    image_hash: u64,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct LatentSizeCacheKey {
    image_hash: u64,
    latent_h: usize,
    latent_w: usize,
}

/// Hash image bytes for cache keying. Uses `DefaultHasher` for speed;
/// collisions are astronomically unlikely at cache sizes <= 256 entries.
/// Not stable across Rust versions — used only for in-process LRU lookup.
pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes);
    hasher.finish()
}

pub(crate) fn prompt_cache_key(prompt: &str, guidance: f64) -> PromptCacheKey {
    PromptCacheKey {
        prompt: prompt.to_string(),
        guidance_bits: guidance.to_bits(),
    }
}

pub(crate) fn prompt_text_key(prompt: &str) -> String {
    prompt.to_string()
}

pub(crate) fn image_size_cache_key(bytes: &[u8], width: u32, height: u32) -> ImageSizeCacheKey {
    ImageSizeCacheKey {
        image_hash: hash_bytes(bytes),
        width,
        height,
    }
}

pub(crate) fn latent_size_cache_key(
    bytes: &[u8],
    latent_h: usize,
    latent_w: usize,
) -> LatentSizeCacheKey {
    LatentSizeCacheKey {
        image_hash: hash_bytes(bytes),
        latent_h,
        latent_w,
    }
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

    pub(crate) fn remove(&mut self, key: &K) {
        self.entries.remove(key);
        self.order.retain(|existing| existing != key);
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

pub(crate) fn restore_cached_tensor<K>(
    cache: &Mutex<LruCache<K, CachedTensor>>,
    key: &K,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>>
where
    K: Eq + Hash + Clone,
{
    restore_or_evict(cache, key, |cached| cached.restore(device, dtype))
}

pub(crate) fn store_cached_tensor<K>(
    cache: &Mutex<LruCache<K, CachedTensor>>,
    key: K,
    tensor: &Tensor,
) -> Result<()>
where
    K: Eq + Hash + Clone,
{
    cache
        .lock()
        .expect("cache poisoned")
        .insert(key, CachedTensor::from_tensor(tensor)?);
    Ok(())
}

pub(crate) fn get_or_insert_cached_tensor<K, F>(
    cache: &Mutex<LruCache<K, CachedTensor>>,
    key: K,
    device: &Device,
    dtype: DType,
    build: F,
) -> Result<(Tensor, bool)>
where
    K: Eq + Hash + Clone,
    F: FnOnce() -> Result<Tensor>,
{
    if let Some(tensor) = restore_cached_tensor(cache, &key, device, dtype)? {
        return Ok((tensor, true));
    }

    let tensor = build()?;
    store_cached_tensor(cache, key, &tensor)?;
    Ok((tensor, false))
}

pub(crate) fn restore_cached_tensor_pair<K>(
    cache: &Mutex<LruCache<K, CachedTensorPair>>,
    key: &K,
    device: &Device,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>>
where
    K: Eq + Hash + Clone,
{
    restore_or_evict(cache, key, |cached| {
        Ok((
            cached.first.restore(device, dtype)?,
            cached.second.restore(device, dtype)?,
        ))
    })
}

pub(crate) fn store_cached_tensor_pair<K>(
    cache: &Mutex<LruCache<K, CachedTensorPair>>,
    key: K,
    first: &Tensor,
    second: &Tensor,
) -> Result<()>
where
    K: Eq + Hash + Clone,
{
    cache
        .lock()
        .expect("cache poisoned")
        .insert(key, CachedTensorPair::from_tensors(first, second)?);
    Ok(())
}

pub(crate) fn get_or_insert_cached_tensor_pair<K, F>(
    cache: &Mutex<LruCache<K, CachedTensorPair>>,
    key: K,
    device: &Device,
    dtype: DType,
    build: F,
) -> Result<((Tensor, Tensor), bool)>
where
    K: Eq + Hash + Clone,
    F: FnOnce() -> Result<(Tensor, Tensor)>,
{
    if let Some((first, second)) = restore_cached_tensor_pair(cache, &key, device, dtype)? {
        return Ok(((first, second), true));
    }

    let (first, second) = build()?;
    store_cached_tensor_pair(cache, key, &first, &second)?;
    Ok(((first, second), false))
}

pub(crate) fn clear_cache<K, V>(cache: &Mutex<LruCache<K, V>>)
where
    K: Eq + Hash + Clone,
{
    cache.lock().expect("cache poisoned").clear();
}

fn restore_or_evict<K, V, T, F>(
    cache: &Mutex<LruCache<K, V>>,
    key: &K,
    restore: F,
) -> Result<Option<T>>
where
    K: Eq + Hash + Clone,
    V: Clone,
    F: FnOnce(V) -> Result<T>,
{
    let cached = cache.lock().expect("cache poisoned").get_cloned(key);
    match cached {
        Some(cached) => match restore(cached) {
            Ok(value) => Ok(Some(value)),
            Err(_) => {
                cache.lock().expect("cache poisoned").remove(key);
                Ok(None)
            }
        },
        None => Ok(None),
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

    #[test]
    fn prompt_cache_key_includes_guidance_bits() {
        assert_ne!(
            prompt_cache_key("hello", 1.0),
            prompt_cache_key("hello", 7.5)
        );
    }

    #[test]
    fn image_size_cache_key_hashes_bytes_and_dimensions() {
        assert_ne!(
            image_size_cache_key(b"abc", 512, 512),
            image_size_cache_key(b"abc", 1024, 1024)
        );
        assert_ne!(
            image_size_cache_key(b"abc", 512, 512),
            image_size_cache_key(b"def", 512, 512)
        );
    }

    #[test]
    fn restore_or_evict_removes_entry_after_restore_failure() {
        let cache = Mutex::new(LruCache::new(1));
        cache.lock().unwrap().insert("a", 1usize);

        let restored: Option<usize> = restore_or_evict(&cache, &"a", |_value| {
            anyhow::bail!("simulated restore failure")
        })
        .unwrap();

        assert!(restored.is_none());
        assert!(cache.lock().unwrap().get_cloned(&"a").is_none());
    }
}
