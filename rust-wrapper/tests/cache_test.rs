/// Integration tests for SCPD KnowledgeCache (store / search / evict / decay)

use adaptive_slm::cache::{KnowledgeCache, CacheConfig};

/// Open an in-memory SQLite cache (`:memory:` path)
async fn open_cache() -> KnowledgeCache {
    KnowledgeCache::new(":memory:").await.expect("failed to open in-memory cache")
}

// ---------------------------------------------------------------------------
// store
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_store_returns_id() {
    let cache = open_cache().await;
    let id = cache.store("rust programming", "Rust is a systems language", 0.5).await.unwrap();
    assert!(id > 0, "store should return a positive row id");
}

#[tokio::test]
async fn test_store_multiple_entries() {
    let cache = open_cache().await;
    let id1 = cache.store("rust", "Rust programming language", 0.5).await.unwrap();
    let id2 = cache.store("python", "Python programming language", 0.5).await.unwrap();
    assert_ne!(id1, id2, "each entry must get a unique id");
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_search_hit() {
    let cache = open_cache().await;
    cache.store("machine learning", "Neural networks and deep learning", 0.5).await.unwrap();

    // Same topic → FNV-1a hash will produce high similarity
    let result = cache.search("machine learning", 0.8).await.unwrap();
    assert!(result.is_some(), "should find an entry for an exact-match query");
}

#[tokio::test]
async fn test_search_miss_high_threshold() {
    let cache = open_cache().await;
    cache.store("cooking recipes", "How to bake bread", 0.5).await.unwrap();

    // Completely unrelated topic at a very high threshold
    let result = cache.search("quantum physics", 0.99).await.unwrap();
    assert!(result.is_none(), "unrelated query at high threshold should miss");
}

#[tokio::test]
async fn test_search_increments_access_count() {
    let cache = open_cache().await;
    let id = cache.store("access test", "Some content here", 0.5).await.unwrap();

    // Access it twice
    let _ = cache.search("access test", 0.5).await.unwrap();
    let _ = cache.search("access test", 0.5).await.unwrap();

    // The entry should have been touched; verify via stats
    let stats = cache.stats().unwrap();
    assert!(stats.total_entries > 0, "cache should have entries");
    assert!(stats.avg_access_count >= 1.0,
        "access_count should reflect hits (got {})", stats.avg_access_count);
    let _ = id;
}

// ---------------------------------------------------------------------------
// eviction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_evict_respects_capacity() {
    let config = CacheConfig { max_entries: 3, ..Default::default() };
    let cache = KnowledgeCache::with_config(":memory:", config).await.unwrap();

    for i in 0..6 {
        cache.store(&format!("topic {}", i), &format!("content {}", i), 0.5).await.unwrap();
    }

    let stats = cache.stats().unwrap();
    assert!(
        stats.total_entries <= 3,
        "cache should have evicted down to max_entries=3, got {}",
        stats.total_entries
    );
}

// ---------------------------------------------------------------------------
// priority decay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_decay_lowers_priority() {
    let cache = open_cache().await;
    let id = cache.store("decay test", "Will decay", 1.0).await.unwrap();

    let stats_before = cache.stats().unwrap();
    cache.decay_and_evict().await.unwrap();
    let stats_after = cache.stats().unwrap();

    assert!(
        stats_after.avg_priority < stats_before.avg_priority,
        "priority should decrease after decay (before={}, after={})",
        stats_before.avg_priority, stats_after.avg_priority
    );
    let _ = id;
}

// ---------------------------------------------------------------------------
// SCPD priority update
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_priority_update_clamps_to_one() {
    let cache = open_cache().await;
    let id = cache.store("priority test", "Test entry", 0.5).await.unwrap();

    // Feed maximum relevance and profile match
    cache.update_priority(id, 1.0, 1.0).await.unwrap();

    let stats = cache.stats().unwrap();
    assert!(
        stats.avg_priority <= 1.0,
        "priority must be clamped to [0,1]"
    );
}
