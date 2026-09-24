//! Tavily Search Integration
//! Industry-standard AI-native search API for RAG

use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::{Result, SLMError};

// ============================================================================
// Types
// ============================================================================

/// Search result from Tavily
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
struct TavilyRequest {
    api_key: String,
    query: String,
    search_depth: String,
    include_answer: bool,
    max_results: u32,
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    answer: Option<String>,
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
    score: f32,
}

// ============================================================================
// Tavily Client
// ============================================================================

/// Tavily API client for AI-optimized web search
pub struct TavilyClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl TavilyClient {
    /// Create a new Tavily client
    pub fn new(api_key: &str) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.to_string(),
            base_url: "https://api.tavily.com".to_string(),
        }
    }
    
    /// Search the web using Tavily
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let request = TavilyRequest {
            api_key: self.api_key.clone(),
            query: query.to_string(),
            search_depth: "basic".to_string(),  // "basic" or "advanced"
            include_answer: true,
            max_results: 5,
        };
        
        let response = self.client
            .post(format!("{}/search", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| SLMError::SearchError(e.to_string()))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(SLMError::SearchError(
                format!("Tavily API error: {} - {}", status, body)
            ));
        }
        
        let tavily_response: TavilyResponse = response.json()
            .await
            .map_err(|e| SLMError::SearchError(e.to_string()))?;
        
        // Convert to our format
        let mut results: Vec<SearchResult> = tavily_response.results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
                score: r.score,
            })
            .collect();
        
        // If Tavily provided a direct answer, add it as the first result
        if let Some(answer) = tavily_response.answer {
            results.insert(0, SearchResult {
                title: "AI Summary".to_string(),
                url: String::new(),
                snippet: answer,
                score: 1.0,
            });
        }
        
        Ok(results)
    }
    
    /// Search with custom options
    pub async fn search_advanced(
        &self,
        query: &str,
        max_results: u32,
        include_domains: Option<&[&str]>,
        exclude_domains: Option<&[&str]>,
    ) -> Result<Vec<SearchResult>> {
        #[derive(Serialize)]
        struct AdvancedRequest {
            api_key: String,
            query: String,
            search_depth: String,
            include_answer: bool,
            max_results: u32,
            #[serde(skip_serializing_if = "Option::is_none")]
            include_domains: Option<Vec<String>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            exclude_domains: Option<Vec<String>>,
        }
        
        let request = AdvancedRequest {
            api_key: self.api_key.clone(),
            query: query.to_string(),
            search_depth: "advanced".to_string(),
            include_answer: true,
            max_results,
            include_domains: include_domains.map(|d| d.iter().map(|s| s.to_string()).collect()),
            exclude_domains: exclude_domains.map(|d| d.iter().map(|s| s.to_string()).collect()),
        };
        
        let response = self.client
            .post(format!("{}/search", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| SLMError::SearchError(e.to_string()))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(SLMError::SearchError(
                format!("Tavily API error: {} - {}", status, body)
            ));
        }
        
        let tavily_response: TavilyResponse = response.json()
            .await
            .map_err(|e| SLMError::SearchError(e.to_string()))?;
        
        Ok(tavily_response.results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
                score: r.score,
            })
            .collect())
    }
}

// ============================================================================
// Fallback: DuckDuckGo (if no API key)
// ============================================================================

/// Free DuckDuckGo search (no API key required)
pub struct DuckDuckGoClient {
    client: Client,
}

impl DuckDuckGoClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
    
    /// Search using DuckDuckGo Instant Answer API
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
            urlencoding::encode(query)
        );
        
        let response = self.client
            .get(&url)
            .header("User-Agent", "AdaptiveSLM/0.1")
            .send()
            .await
            .map_err(|e| SLMError::SearchError(e.to_string()))?;
        
        #[derive(Deserialize)]
        struct DDGResponse {
            #[serde(rename = "AbstractText")]
            abstract_text: Option<String>,
            #[serde(rename = "AbstractURL")]
            abstract_url: Option<String>,
            #[serde(rename = "RelatedTopics")]
            related_topics: Vec<DDGTopic>,
        }
        
        #[derive(Deserialize)]
        struct DDGTopic {
            #[serde(rename = "Text")]
            text: Option<String>,
            #[serde(rename = "FirstURL")]
            first_url: Option<String>,
        }
        
        let ddg: DDGResponse = response.json()
            .await
            .map_err(|e| SLMError::SearchError(e.to_string()))?;
        
        let mut results = Vec::new();
        
        // Add abstract if available
        if let (Some(text), Some(url)) = (ddg.abstract_text, ddg.abstract_url) {
            if !text.is_empty() {
                results.push(SearchResult {
                    title: "Summary".to_string(),
                    url,
                    snippet: text,
                    score: 1.0,
                });
            }
        }
        
        // Add related topics
        for topic in ddg.related_topics.into_iter().take(5) {
            if let (Some(text), Some(url)) = (topic.text, topic.first_url) {
                results.push(SearchResult {
                    title: text.chars().take(50).collect::<String>() + "...",
                    url,
                    snippet: text,
                    score: 0.8,
                });
            }
        }
        
        Ok(results)
    }
}

impl Default for DuckDuckGoClient {
    fn default() -> Self {
        Self::new()
    }
}
