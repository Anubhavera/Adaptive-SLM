//! Elastic model-family selector — ties PAKD profiles to MatFormer-style depth.
//!
//! AdaptiveSLM v2 exports one GGUF per elastic exit depth (30L / 22L / 15L)
//! from a single training run. The PAKD policy maps user expertise to the
//! smallest depth that still serves the user:
//!
//! * Beginner      → 15L
//! * Intermediate  → 22L
//! * Advanced/Expert → 30L
//!
//! File name pattern: `adaptive_slm-{15l|22l|30l}-q4_k_m.gguf`.

use std::path::{Path, PathBuf};

use crate::profile::Expertise;

/// A depth variant of the AdaptiveSLM model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DepthChoice {
    D15,
    D22,
    D30,
}

impl DepthChoice {
    /// Depth tag used in the GGUF file name (e.g. `adaptive_slm-15l-…`).
    pub fn tag(self) -> &'static str {
        match self {
            DepthChoice::D15 => "15l",
            DepthChoice::D22 => "22l",
            DepthChoice::D30 => "30l",
        }
    }
}

/// Depths ordered small → large (fallback scans this order).
const ALL_DEPTHS: [DepthChoice; 3] = [DepthChoice::D15, DepthChoice::D22, DepthChoice::D30];

/// Directory-backed selector for the elastic model family.
#[derive(Debug, Clone)]
pub struct ModelFamily {
    /// Directory containing the family GGUF files.
    pub dir: PathBuf,
}

impl ModelFamily {
    /// Create a selector over `dir`.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }

    fn model_path(&self, depth: DepthChoice) -> PathBuf {
        self.dir
            .join(format!("adaptive_slm-{}-q4_k_m.gguf", depth.tag()))
    }

    /// Preferred depth for an expertise level (PAKD policy).
    pub fn preferred_depth(expertise: Expertise) -> DepthChoice {
        match expertise {
            Expertise::Beginner => DepthChoice::D15,
            Expertise::Intermediate => DepthChoice::D22,
            Expertise::Advanced | Expertise::Expert => DepthChoice::D30,
        }
    }

    /// Pick the GGUF to run for this expertise level.
    ///
    /// Policy: the preferred depth, else the next larger available depth,
    /// else any `*.gguf` in the directory. `None` if the directory holds no
    /// GGUF files.
    pub fn pick(&self, expertise: Expertise) -> Option<PathBuf> {
        let preferred = Self::preferred_depth(expertise);

        for depth in ALL_DEPTHS.into_iter().filter(|d| *d >= preferred) {
            let path = self.model_path(depth);
            if path.is_file() {
                return Some(path);
            }
        }

        let mut any: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().map_or(false, |ext| ext == "gguf")
            })
            .collect();
        any.sort();
        any.into_iter().next()
    }

    /// Family depths present in the directory, ordered small → large.
    pub fn available_depths(&self) -> Vec<DepthChoice> {
        ALL_DEPTHS
            .into_iter()
            .filter(|depth| self.model_path(*depth).is_file())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("aslm_family_test_{}_{}", std::process::id(), name));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("failed to create temp dir");
            Self(dir)
        }

        fn touch(&self, file: &str) -> PathBuf {
            let path = self.0.join(file);
            std::fs::write(&path, b"").expect("failed to create dummy file");
            path
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn pick_maps_expertise_to_depth_when_all_present() {
        let dir = TempDir::new("all_depths");
        dir.touch("adaptive_slm-15l-q4_k_m.gguf");
        dir.touch("adaptive_slm-22l-q4_k_m.gguf");
        dir.touch("adaptive_slm-30l-q4_k_m.gguf");
        let family = ModelFamily::new(dir.path());

        assert_eq!(
            family.pick(Expertise::Beginner),
            Some(dir.path().join("adaptive_slm-15l-q4_k_m.gguf"))
        );
        assert_eq!(
            family.pick(Expertise::Intermediate),
            Some(dir.path().join("adaptive_slm-22l-q4_k_m.gguf"))
        );
        assert_eq!(
            family.pick(Expertise::Advanced),
            Some(dir.path().join("adaptive_slm-30l-q4_k_m.gguf"))
        );
        assert_eq!(
            family.pick(Expertise::Expert),
            Some(dir.path().join("adaptive_slm-30l-q4_k_m.gguf"))
        );
    }

    #[test]
    fn pick_falls_back_to_next_larger_depth() {
        let dir = TempDir::new("larger_fallback");
        dir.touch("adaptive_slm-22l-q4_k_m.gguf");
        dir.touch("adaptive_slm-30l-q4_k_m.gguf");
        let family = ModelFamily::new(dir.path());

        assert_eq!(
            family.pick(Expertise::Beginner),
            Some(dir.path().join("adaptive_slm-22l-q4_k_m.gguf"))
        );
        assert_eq!(
            family.pick(Expertise::Intermediate),
            Some(dir.path().join("adaptive_slm-22l-q4_k_m.gguf"))
        );
        assert_eq!(
            family.pick(Expertise::Expert),
            Some(dir.path().join("adaptive_slm-30l-q4_k_m.gguf"))
        );
    }

    #[test]
    fn pick_falls_back_to_only_largest() {
        let dir = TempDir::new("only_30l");
        dir.touch("adaptive_slm-30l-q4_k_m.gguf");
        let family = ModelFamily::new(dir.path());

        for expertise in [
            Expertise::Beginner,
            Expertise::Intermediate,
            Expertise::Advanced,
        ] {
            assert_eq!(
                family.pick(expertise),
                Some(dir.path().join("adaptive_slm-30l-q4_k_m.gguf")),
                "expected 30l fallback for {:?}",
                expertise
            );
        }
    }

    #[test]
    fn pick_falls_back_to_any_gguf() {
        let dir = TempDir::new("any_gguf");
        dir.touch("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        let family = ModelFamily::new(dir.path());

        assert_eq!(
            family.pick(Expertise::Advanced),
            Some(dir.path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        );
    }

    #[test]
    fn pick_uses_smaller_family_model_when_no_larger_exists() {
        let dir = TempDir::new("smaller_via_any");
        dir.touch("adaptive_slm-15l-q4_k_m.gguf");
        let family = ModelFamily::new(dir.path());

        assert_eq!(
            family.pick(Expertise::Expert),
            Some(dir.path().join("adaptive_slm-15l-q4_k_m.gguf"))
        );
    }

    #[test]
    fn pick_returns_none_for_empty_or_missing_dir() {
        let dir = TempDir::new("empty");
        let family = ModelFamily::new(dir.path());
        assert_eq!(family.pick(Expertise::Beginner), None);

        let missing = ModelFamily::new(dir.path().join("does_not_exist"));
        assert_eq!(missing.pick(Expertise::Expert), None);
    }

    #[test]
    fn available_depths_lists_present_depths_sorted() {
        let dir = TempDir::new("depths");
        dir.touch("adaptive_slm-30l-q4_k_m.gguf");
        dir.touch("adaptive_slm-15l-q4_k_m.gguf");
        dir.touch("unrelated.txt");
        let family = ModelFamily::new(dir.path());

        assert_eq!(
            family.available_depths(),
            vec![DepthChoice::D15, DepthChoice::D30]
        );

        let empty = TempDir::new("depths_empty");
        let empty_family = ModelFamily::new(empty.path());
        assert!(empty_family.available_depths().is_empty());
    }

    #[test]
    fn non_gguf_files_are_ignored() {
        let dir = TempDir::new("non_gguf");
        dir.touch("adaptive_slm-30l-q4_k_m.gguf.txt");
        let family = ModelFamily::new(dir.path());
        assert_eq!(family.pick(Expertise::Expert), None);
    }
}
