-- AdaptiveSLM Storage Schema
-- Uses sqlite-vec for vector similarity search

-- User profiles
CREATE TABLE IF NOT EXISTS user_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    age INTEGER NOT NULL,
    background TEXT,
    expertise_level INTEGER DEFAULT 1,  -- 0=beginner, 1=intermediate, 2=advanced, 3=expert
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- User interests (many-to-one)
CREATE TABLE IF NOT EXISTS user_interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    interest TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interests_profile ON user_interests(profile_id);

-- Knowledge cache (SCPD)
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,                     -- 384-dim float32 vector
    priority_score REAL DEFAULT 0.5,    -- SCPD priority [0, 1]
    access_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    last_accessed TEXT DEFAULT (datetime('now')),
    source_url TEXT,
    source_type TEXT DEFAULT 'generated'  -- 'generated', 'web', 'user'
);

CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_priority ON knowledge(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_accessed ON knowledge(last_accessed DESC);

-- Virtual table for vector search (requires sqlite-vec extension)
-- Uncomment when sqlite-vec is installed:
-- CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0(
--     embedding float[384]
-- );

-- Query history for relevance tracking
CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    profile_id INTEGER REFERENCES user_profiles(id),
    cache_hit BOOLEAN DEFAULT 0,
    response_time_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_query_created ON query_history(created_at DESC);

-- ACC (Adaptive Context Compression) telemetry
CREATE TABLE IF NOT EXISTS acc_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    available_ram_mb INTEGER,
    total_ram_mb INTEGER,
    cpu_usage REAL,
    battery_level REAL,
    context_size INTEGER,
    timestamp TEXT DEFAULT (datetime('now'))
);

-- Scheduled maintenance: decay priorities daily
-- Run: UPDATE knowledge SET priority_score = priority_score * 0.95 WHERE priority_score > 0.1;
-- Run: DELETE FROM knowledge WHERE priority_score < 0.05 AND access_count < 2;
