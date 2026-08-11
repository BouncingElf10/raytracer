//! Composes the path-tracing compute shader from its WGSL fragments.
//!
//! The instrumented and clean variants come from one source (§4): instrumentation
//! lives inside `//#if INSTRUMENTED` blocks that this module either keeps or
//! strips before handing the text to naga. Stripping happens at the text level,
//! so the clean variant provably contains no counter declarations, no counter
//! buffer binding and no counter writes -- nothing for the driver to keep alive
//! and nothing that could skew the timed pass.

const TYPES: &str = include_str!("shaders/types.wgsl");
const BINDINGS: &str = include_str!("shaders/bindings.wgsl");
const HIT: &str = include_str!("shaders/hit.wgsl");
const RANDOM: &str = include_str!("shaders/random.wgsl");
const PATH: &str = include_str!("shaders/path.wgsl");
const ENTRY: &str = include_str!("shaders/raytracer.wgsl");

/// Which build of the traversal shader to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderVariant {
    /// No instrumentation whatsoever. Used for the interactive renderer and for
    /// the timed pass (§6 Pass B).
    Clean,
    /// Writes per-ray traversal counters to binding 8 (§6 Pass A).
    Instrumented,
}

impl ShaderVariant {
    fn instrumented(self) -> bool {
        matches!(self, ShaderVariant::Instrumented)
    }
}

/// Full WGSL source for the requested variant.
pub fn compose(variant: ShaderVariant) -> String {
    let fragments = [TYPES, BINDINGS, RANDOM, HIT, PATH, ENTRY];
    let mut out = String::new();
    for fragment in fragments {
        out.push_str(&preprocess(fragment, variant.instrumented()));
        out.push('\n');
    }
    out
}

/// Minimal line-oriented conditional preprocessor.
///
/// Understands `//#if INSTRUMENTED`, `//#else` and `//#endif`. Removed lines are
/// replaced by blank lines rather than deleted, so both variants stay
/// line-for-line aligned and a naga error message points at the same place in
/// either build.
fn preprocess(source: &str, instrumented: bool) -> String {
    let mut out = String::with_capacity(source.len());
    // One entry per open `//#if`: whether its body should be emitted.
    let mut branches: Vec<bool> = Vec::new();

    for line in source.lines() {
        let directive = line.trim();
        let emitting = branches.iter().all(|keep| *keep);

        if let Some(condition) = directive.strip_prefix("//#if ") {
            let taken = match condition.trim() {
                "INSTRUMENTED" => instrumented,
                "!INSTRUMENTED" => !instrumented,
                other => panic!("unknown shader preprocessor condition: {other:?}"),
            };
            branches.push(taken);
        } else if directive == "//#else" {
            let taken = branches
                .last_mut()
                .expect("//#else without a matching //#if");
            *taken = !*taken;
        } else if directive == "//#endif" {
            branches.pop().expect("//#endif without a matching //#if");
        } else if emitting {
            out.push_str(line);
        }

        out.push('\n');
    }

    assert!(branches.is_empty(), "unterminated //#if in shader source");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_instrumentation_from_the_clean_variant() {
        let clean = compose(ShaderVariant::Clean);
        assert!(!clean.contains("ray_counters"));
        assert!(!clean.contains("ctr_node_visits"));
        assert!(!clean.contains("binding(8)"));
        // `atomic` also appears in prose, so match the WGSL type constructor.
        assert!(!clean.contains("atomic<"));
    }

    #[test]
    fn keeps_instrumentation_in_the_instrumented_variant() {
        let instrumented = compose(ShaderVariant::Instrumented);
        assert!(instrumented.contains("ray_counters"));
        assert!(instrumented.contains("ctr_prim_tests"));
        assert!(instrumented.contains("binding(8)"));
    }

    #[test]
    fn variants_stay_line_aligned() {
        let clean = compose(ShaderVariant::Clean).lines().count();
        let instrumented = compose(ShaderVariant::Instrumented).lines().count();
        assert_eq!(clean, instrumented);
    }
}
