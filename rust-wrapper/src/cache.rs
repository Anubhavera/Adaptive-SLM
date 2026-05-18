//! Semantic Cache with Priority Decay (SCPD)
//! Novel Contribution #3
//! 
//! Intelligent knowledge retention combining:
//! - Access recency (LRU component)
//! - Semantic relevance to recent queries
//! - User profile alignment

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::path::Path;
use crate::{Result, SLMError};

// ============================================================================
// Types
// ============================================================================

/// A cached knowledge entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub id: i64,
    pub topic: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub priority_score: f32,
    pub access_count: i32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub source_url: Option<String>,
}

/// SCPD Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries in cache
    pub max_entries: usize,
    /// Recency weight (α)
    pub recency_weight: f32,
    /// Relevance weight (β)
    pub relevance_weight: f32,
    /// Profile match weight (γ)
    pub profile_weight: f32,
    /// Decay factor per day
    pub decay_factor: f32,
    /// Minimum priority before eviction
    pub eviction_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            recency_weight: 0.4,
            relevance_weight: 0.35,
            profile_weight: 0.25,
            decay_factor: 0.95,
            eviction_threshold: 0.1,
        }
    }
}

// ============================================================================
// Knowledge Cache Implementation
// ============================================================================

/// SCPD-based knowledge cache using SQLite + sqlite-vec
pub struct KnowledgeCache {
    conn: Connection,
    config: CacheConfig,
}

impl KnowledgeCache {
    /// Create or open a cache database
    pub async fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        // Create tables
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                priority_score REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                source_url TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_topic ON knowledge(topic);
            CREATE INDEX IF NOT EXISTS idx_priority ON knowledge(priority_score DESC);
            CREATE INDEX IF NOT EXISTS idx_accessed ON knowledge(last_accessed DESC);
        "#).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        // Try to load sqlite-vec extension (if available)
        // Note: This requires the extension to be installed
        let _ = conn.load_extension_enable();
        if let Err(e) = conn.load_extension(Path::new("vec0"), None) {
            tracing::warn!("sqlite-vec not available, using fallback: {}", e);
            // Fallback: we'll do vector search in Rust
        }
        
        Ok(Self {
            conn,
            config: CacheConfig::default(),
        })
    }
    
    /// Store knowledge with initial priority
    pub async fn store(
        &self,
        topic: &str,
        content: &str,
        initial_priority: f32,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        
        // Generate embedding (simplified - in production, use the embedding engine)
        let embedding = self.compute_embedding(topic);
        let embedding_bytes = bytemuck_cast(&embedding);
        
        self.conn.execute(
            r#"
            INSERT INTO knowledge (topic, content, embedding, priority_score, created_at, last_accessed)
            VALUES (?1, ?2, ?3, ?4, ?5, ?5)
            "#,
            params![topic, content, embedding_bytes, initial_priority, now],
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        let id = self.conn.last_insert_rowid();
        
        // Evict if over capacity
        self.evict_if_needed().await?;
        
        Ok(id)
    }
    
    /// Search cache by semantic similarity
    pub async fn search(
        &self,
        query: &str,
        min_similarity: f32,
    ) -> Result<Option<KnowledgeEntry>> {
        let query_embedding = self.compute_embedding(query);
        
        // Fetch all entries and compute similarity
        // In production with sqlite-vec, this would be a native vector search
        let mut stmt = self.conn.prepare(
            "SELECT id, topic, content, embedding, priority_score, access_count, created_at, last_accessed, source_url FROM knowledge"
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        let entries = stmt.query_map([], |row| {
            let embedding_bytes: Vec<u8> = row.get(3)?;
            let embedding = bytemuck_from(&embedding_bytes);
            
            Ok(KnowledgeEntry {
                id: row.get(0)?,
                topic: row.get(1)?,
                content: row.get(2)?,
                embedding,
                priority_score: row.get(4)?,
                access_count: row.get(5)?,
                created_at: row.get::<_, String>(6)?.parse().unwrap_or_else(|_| Utc::now()),
                last_accessed: row.get::<_, String>(7)?.parse().unwrap_or_else(|_| Utc::now()),
                source_url: row.get(8).ok(),
            })
        }).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        let mut best_match: Option<(f32, KnowledgeEntry)> = None;
        
        for entry_result in entries {
            if let Ok(entry) = entry_result {
                let sim = cosine_similarity(&query_embedding, &entry.embedding);
                
                if sim >= min_similarity {
                    if best_match.is_none() || sim > best_match.as_ref().unwrap().0 {
                        best_match = Some((sim, entry));
                    }
                }
            }
        }
        
        if let Some((_, mut entry)) = best_match {
            // Update access tracking
            self.update_access(entry.id).await?;
            entry.access_count += 1;
            entry.last_accessed = Utc::now();
            return Ok(Some(entry));
        }
        
        Ok(None)
    }
    
    /// Update access count and recency
    async fn update_access(&self, id: i64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        
        self.conn.execute(
            "UPDATE knowledge SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
            params![now, id],
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Apply priority decay and evict low-priority entries
    pub async fn decay_and_evict(&self) -> Result<usize> {
        // Apply decay to all entries
        self.conn.execute(
            "UPDATE knowledge SET priority_score = priority_score * ?1",
            params![self.config.decay_factor],
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        self.evict_if_needed().await
    }
    
    /// Evict entries if over capacity or below threshold
    async fn evict_if_needed(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge",
            [],
            |row| row.get(0),
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        if count as usize <= self.config.max_entries {
            return Ok(0);
        }
        
        // Evict lowest priority entries
        let to_evict = count as usize - self.config.max_entries;
        
        self.conn.execute(
            r#"
            DELETE FROM knowledge WHERE id IN (
                SELECT id FROM knowledge
                ORDER BY priority_score ASC
                LIMIT ?1
            )
            "#,
            params![to_evict as i64],
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        Ok(to_evict)
    }
    
    /// Update priority based on SCPD formula
    pub async fn update_priority(
        &self,
        id: i64,
        relevance: f32,
        profile_match: f32,
    ) -> Result<()> {
        // Fetch current entry
        let (access_count, last_accessed): (i32, String) = self.conn.query_row(
            "SELECT access_count, last_accessed FROM knowledge WHERE id = ?1",
            params![id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        // Compute recency score
        let last: DateTime<Utc> = last_accessed.parse().unwrap_or_else(|_| Utc::now());
        let days_ago = (Utc::now() - last).num_days() as f32;
        let recency = (-0.1 * days_ago).exp(); // Exponential decay
        
        // SCPD formula: Priority = α·recency + β·relevance + γ·profile_match
        let priority = self.config.recency_weight * recency
            + self.config.relevance_weight * relevance
            + self.config.profile_weight * profile_match;
        
        // Boost by access frequency (log scale)
        let access_boost = 1.0 + 0.1 * (access_count as f32 + 1.0).ln();
        let final_priority = (priority * access_boost).clamp(0.0, 1.0);
        
        self.conn.execute(
            "UPDATE knowledge SET priority_score = ?1 WHERE id = ?2",
            params![final_priority, id],
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> Result<CacheStats> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge",
            [],
            |row| row.get(0),
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        let avg_priority: f32 = self.conn.query_row(
            "SELECT COALESCE(AVG(priority_score), 0) FROM knowledge",
            [],
            |row| row.get(0),
        ).map_err(|e| SLMError::CacheError(e.to_string()))?;
        
        Ok(CacheStats {
            entry_count: count as usize,
            avg_priority,
            max_entries: self.config.max_entries,
        })
    }
    
    /// Compute text embedding using word + character n-gram feature hashing.
    /// Produces a 384-dim vector where semantically similar texts cluster together.
    /// This is a lightweight alternative to neural embeddings, suitable for on-device
    /// cache similarity matching without requiring an external embedding model.
    fn compute_embedding(&self, text: &str) -> Vec<f32> {
        let dim = 384usize;
        let mut embedding = vec![0.0f32; dim];
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Word unigram features (captures topic keywords)
        for word in &words {
            let h = Self::fnv1a_hash(word.as_bytes());
            let idx = (h as usize) % dim;
            let sign = if (h >> 17) & 1 == 0 { 1.0 } else { -1.0 };
            embedding[idx] += sign;
        }

        // Word bigram features (captures phrasing/context)
        for pair in words.windows(2) {
            let combined = format!("{} {}", pair[0], pair[1]);
            let h = Self::fnv1a_hash(combined.as_bytes());
            let idx = (h as usize) % dim;
            let sign = if (h >> 17) & 1 == 0 { 1.0 } else { -1.0 };
            embedding[idx] += sign * 0.7;
        }

        // Character trigram features (captures morphology, handles typos)
        let chars: Vec<char> = lower.chars().collect();
        for window in chars.windows(3) {
            let trigram: String = window.iter().collect();
            let h = Self::fnv1a_hash(trigram.as_bytes());
            let idx = (h as usize) % dim;
            let sign = if (h >> 17) & 1 == 0 { 1.0 } else { -1.0 };
            embedding[idx] += sign * 0.3;
        }

        normalize(&mut embedding);
        embedding
    }

    /// FNV-1a hash for feature hashing — fast, good distribution
    fn fnv1a_hash(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub entry_count: usize,
    pub avg_priority: f32,
    pub max_entries: usize,
}

// ============================================================================
// Vector Utilities
// ============================================================================

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn bytemuck_cast(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytemuck_from(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
