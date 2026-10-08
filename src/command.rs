//! Pure native command identities, help and the existing key/value grammar.
use crate::{require, Result};
use std::collections::BTreeMap;

// The public/comparison groups generate their ordered help lists. Private IDs
// retain their legacy startup class and remain absent from the public help.
macro_rules! native_commands {
    (
        public: [$first:ident => $first_id:literal $(, $rest:ident => $rest_id:literal)* $(,)?],
        comparison: [$compare_first:ident => $compare_first_id:literal $(, $compare_rest:ident => $compare_rest_id:literal)* $(,)?],
        private: [$($private:ident => $private_id:literal),+ $(,)?],
        intake: { cache: $intake_cache:literal, verify_sha: $intake_verify_sha:literal },
        serve: { port: $serve_port:literal },
        comparison_defaults: { cache: $comparison_cache:literal, tensor: $comparison_tensor:literal, row: $comparison_row:literal, col: $comparison_col:literal, quantity: $comparison_quantity:literal, mapping: $comparison_mapping:literal, x: $comparison_x:literal, y: $comparison_y:literal, port: $comparison_port:literal },
        rendering: {
            overview: { slice: $overview_slice:literal, rules: $overview_rules:literal, max_values: $overview_max:literal },
            tile: { tensor: $tile_tensor:literal, slice: $tile_slice:literal, rules: $tile_rules:literal, x: $tile_x:literal, y: $tile_y:literal, out: $tile_out:literal },
            bench: { tensor: $bench_tensor:literal, repeats: $bench_repeats:literal }
        },
        help_parts: [$help0:literal, $help1:literal, $help2:literal, $help3:literal, $help4:literal, $help5:literal, $help6:literal, $help7:literal] $(,)?
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Command {
            $first,
            $($rest,)*
            $compare_first,
            $($compare_rest,)*
            $($private,)*
        }

        impl Command {
            /// Resolve exact native IDs; unknown spelling retains caller behavior.
            pub fn lookup(id: &str) -> Option<Self> {
                match id {
                    $first_id => Some(Self::$first),
                    $($rest_id => Some(Self::$rest),)*
                    $compare_first_id => Some(Self::$compare_first),
                    $($compare_rest_id => Some(Self::$compare_rest),)*
                    $($private_id => Some(Self::$private),)*
                    _ => None,
                }
            }

            /// Public viewer/reader commands use the existing standalone policy.
            pub fn is_standalone(self) -> bool {
                matches!(self, Self::$first $(| Self::$rest)*)
            }
        }

        /// Command-specific fallback facts; they do not insert or validate options.
        pub mod defaults {
            pub const INTAKE_CACHE: &str = $intake_cache;
            pub const INTAKE_VERIFY_SHA: &str = $intake_verify_sha;
            pub const SERVE_PORT: &str = $serve_port;
            pub const COMPARISON_CACHE: &str = $comparison_cache;
            pub const COMPARISON_TENSOR: &str = $comparison_tensor;
            pub const COMPARISON_ROW: &str = $comparison_row;
            pub const COMPARISON_COL: &str = $comparison_col;
            pub const COMPARISON_QUANTITY: &str = $comparison_quantity;
            pub const COMPARISON_MAPPING: &str = $comparison_mapping;
            pub const COMPARISON_X: &str = $comparison_x;
            pub const COMPARISON_Y: &str = $comparison_y;
            pub const COMPARISON_PORT: &str = $comparison_port;
            pub const OVERVIEW_SLICE: &str = $overview_slice;
            pub const OVERVIEW_RULES: &str = $overview_rules;
            pub const OVERVIEW_MAX_VALUES: &str = $overview_max;
            pub const TILE_TENSOR: &str = $tile_tensor;
            pub const TILE_SLICE: &str = $tile_slice;
            pub const TILE_RULES: &str = $tile_rules;
            pub const TILE_X: &str = $tile_x;
            pub const TILE_Y: &str = $tile_y;
            pub const TILE_OUT: &str = $tile_out;
            pub const BENCH_TENSOR: &str = $bench_tensor;
            pub const BENCH_REPEATS: &str = $bench_repeats;
        }

        pub const HELP: &str = concat!(
            "Weight Atlas Rust\nCommands: ",
            $first_id, $(" | ", $rest_id,)*
            "\nComparison: ",
            $compare_first_id, $(" | ", $compare_rest_id,)*
            $help0, $intake_cache, $help1, $serve_port, $help2,
            $intake_verify_sha, $help3, $overview_rules, $help4,
            $overview_max, $help5, $tile_rules, $help6, $bench_repeats, $help7
        );
    };
}

native_commands! {
    public: [
        Metadata => "metadata",
        Serve => "serve",
        Calibrate => "calibrate",
        Verify => "verify",
        Tile => "tile",
        Overview => "overview",
        Inspect => "inspect",
        Bench => "bench",
    ],
    comparison: [
        CompareMetadata => "compare-metadata",
        CompareCalibrate => "compare-calibrate",
        CompareTile => "compare-tile",
        CompareInspect => "compare-inspect",
        CompareServe => "compare-serve",
    ],
    private: [
        HostedRenderer => "hosted-renderer",
        ProfileWorker => "profile-worker",
    ],
    intake: { cache: "cache", verify_sha: "false" },
    serve: { port: "8775" },
    comparison_defaults: { cache: "cache-comparison", tensor: "0", row: "0", col: "0", quantity: "delta", mapping: "linear", x: "0", y: "0", port: "8776" },
    rendering: {
        overview: { slice: "", rules: "tensor_linear,tensor_asinh", max_values: "16777216" },
        tile: { tensor: "0", slice: "", rules: "global_linear,global_asinh", x: "0", y: "0", out: "tile" },
        bench: { tensor: "0", repeats: "3" }
    },
    help_parts: [
        "; explicit --model A --compare-model B; --cache outside both\nComparison tile: --tensor ID --quantity a|b|delta|abs_delta --mapping linear|asinh|magnitude --out PREFIX; compare-calibrate requires --tensor ID\nRequired: --model DIRECTORY\nCommon: --cache DIRECTORY (default ./",
        ") --name NAME --revision REVISION\nserve: --port ",
        " --verify-sha ",
        "; true hashes all bytes before listening; no remote binding\ncalibrate: --tensor ID (omit for all); resumes valid calibration\nverify: full SHA-256 reads, compares local model-api.json if present\noverview: --tensor ID --slice INDICES --rules ",
        " --max-values ",
        "; explicit bounded coarse preparation\ntile: --tensor ID --rules ",
        " --level L --x X --y Y --out PREFIX\ninspect: --tensor ID --row R --col C\nbench: --tensor ID --repeats ",
        "; factors 1,4,16, both global rules\nDefaults: one CPU, 768 MiB address-space budget, >=3 GiB effective RAM reserve plus process budget, >=25 GiB disk reserve.\n--resources JSON selects validated standalone budgets; see docs/RESOURCES.md.\nCache identity checks do not claim a fresh full-content hash."
    ],
}

/// Parse key/value arguments after a present command token, in original order.
pub fn parse_options(args: &[String]) -> Result<BTreeMap<String, String>> {
    let mut opts = BTreeMap::new();
    require((args.len() - 1).is_multiple_of(2), "Options require values")?;
    for p in args[1..].chunks_exact(2) {
        require(p[0].starts_with("--"), "Expected --option value")?;
        require(
            opts.insert(p[0][2..].to_owned(), p[1].clone()).is_none(),
            "Duplicate CLI option",
        )?
    }
    Ok(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_retains_public_comparison_and_private_ids() {
        for (id, expected, standalone) in [
            ("metadata", Command::Metadata, true),
            ("serve", Command::Serve, true),
            ("calibrate", Command::Calibrate, true),
            ("verify", Command::Verify, true),
            ("tile", Command::Tile, true),
            ("overview", Command::Overview, true),
            ("inspect", Command::Inspect, true),
            ("bench", Command::Bench, true),
            ("compare-metadata", Command::CompareMetadata, false),
            ("compare-calibrate", Command::CompareCalibrate, false),
            ("compare-tile", Command::CompareTile, false),
            ("compare-inspect", Command::CompareInspect, false),
            ("compare-serve", Command::CompareServe, false),
            ("hosted-renderer", Command::HostedRenderer, false),
            ("profile-worker", Command::ProfileWorker, false),
        ] {
            assert_eq!(Command::lookup(id), Some(expected));
            assert_eq!(expected.is_standalone(), standalone);
        }
        for id in ["", "Metadata", "unknown", "compare-unknown", "--help"] {
            assert_eq!(Command::lookup(id), None);
        }
    }
}

/// Legacy unknown-prefix routing remains separate from exact command lookup.
pub const COMPARISON_PREFIX: &str = "compare-";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub common: &'static [&'static str],
    pub specific: &'static [&'static str],
}

pub mod options {
    // Common documented arguments stay known even when an early command ignores
    // them. Existing resource-policy refusal remains separate from key spelling.
    pub const COMMON: &[&str] = &["model", "cache", "name", "revision", "resources"];
    pub const PROFILE_REQUIRED: &[&str] = &[
        "model",
        "revision",
        "tensor",
        "slice",
        "seed",
        "values",
        "wall-ms",
        "cpu-ms",
        "binding",
        "output-fd",
    ];
    pub const PROFILE_RESTORE: &[&str] = &["input-fd", "input-sha"];
}

impl Command {
    /// Audited option names; ordinary command intake remains permissive here.
    pub fn options(self) -> Options {
        use crate::api::parameter::{
            COL, LEFT, LEVEL, MAPPING, QUANTITY, RIGHT, ROW, SLICE, TENSOR, X, Y,
        };
        let specific: &[&str] = match self {
            Self::Metadata | Self::Verify => &[],
            Self::Serve => &["port", "verify-sha"],
            Self::Calibrate => &[TENSOR],
            Self::Overview => &[TENSOR, SLICE, "rules", "max-values"],
            Self::Tile => &[TENSOR, SLICE, "rules", LEVEL, X, Y, "out"],
            Self::Inspect => &[TENSOR, SLICE, ROW, COL, LEFT, RIGHT],
            Self::Bench => &[TENSOR, "repeats"],
            Self::CompareMetadata | Self::CompareCalibrate => &["compare-model", TENSOR],
            Self::CompareServe => &["compare-model", TENSOR, "port"],
            Self::CompareTile => &[
                "compare-model",
                TENSOR,
                QUANTITY,
                MAPPING,
                LEVEL,
                X,
                Y,
                "out",
            ],
            Self::CompareInspect => &["compare-model", TENSOR, ROW, COL],
            Self::HostedRenderer => &["channel-fd"],
            Self::ProfileWorker => options::PROFILE_RESTORE,
        };
        Options {
            common: if self == Self::ProfileWorker {
                options::PROFILE_REQUIRED
            } else {
                options::COMMON
            },
            specific,
        }
    }
}

/// Typed option facts. Reading a field never validates unrelated/unknown names.
/// The existing help/default declaration above remains the source of fallback text.
pub mod argument {
    use super::defaults;
    use crate::parameter::parse_indices;
    use crate::parameter::{number, text, Parameter};

    macro_rules! arguments {
        ($($id:ident: $ty:ty = ($name:expr, $default:expr, $missing:expr, $parser:expr);)*) => {
            $(pub const $id: Parameter<$ty> = Parameter::new($name, $default, $missing, $parser);)*
        };
    }

    arguments! {
        MODEL: String = ("model", "", Some("--model DIRECTORY is required"), text);
        CACHE: String = ("cache", defaults::INTAKE_CACHE, None, text);
        NAME: String = ("name", "", None, text);
        REVISION: String = ("revision", "", None, text);
        RESOURCES: String = ("resources", "", None, text);
        SERVE_PORT: u16 = ("port", defaults::SERVE_PORT, None, number);
        VERIFY_SHA: String = ("verify-sha", defaults::INTAKE_VERIFY_SHA, None, text);
        CALIBRATE_TENSOR: usize = ("tensor", "", None, number);
        CHANNEL_FD: i32 = ("channel-fd", "", Some("Private channel required"), number);
        OVERVIEW_TENSOR: usize = ("tensor", "", Some("Overview requires explicit --tensor ID"), number);
        OVERVIEW_SLICE: Vec<usize> = ("slice", defaults::OVERVIEW_SLICE, None, parse_indices);
        OVERVIEW_RULES: String = ("rules", defaults::OVERVIEW_RULES, None, text);
        OVERVIEW_MAX_VALUES: usize = ("max-values", defaults::OVERVIEW_MAX_VALUES, None, number);
        TILE_TENSOR: usize = ("tensor", defaults::TILE_TENSOR, None, number);
        TILE_SLICE: Vec<usize> = ("slice", defaults::TILE_SLICE, None, parse_indices);
        TILE_RULES: String = ("rules", defaults::TILE_RULES, None, text);
        TILE_LEVEL: u32 = ("level", "", None, number);
        TILE_X: usize = ("x", defaults::TILE_X, None, number);
        TILE_Y: usize = ("y", defaults::TILE_Y, None, number);
        TILE_OUT: String = ("out", defaults::TILE_OUT, None, text);
        BENCH_TENSOR: usize = ("tensor", defaults::BENCH_TENSOR, None, number);
        BENCH_REPEATS: usize = ("repeats", defaults::BENCH_REPEATS, None, number);
        COMPARE_MODEL: String = ("compare-model", "", Some("--compare-model DIRECTORY is required for comparison"), text);
        COMPARE_CACHE: String = ("cache", defaults::COMPARISON_CACHE, None, text);
        COMPARE_TENSOR: usize = ("tensor", defaults::COMPARISON_TENSOR, None, number);
        COMPARE_ROW: usize = ("row", defaults::COMPARISON_ROW, None, number);
        COMPARE_COL: usize = ("col", defaults::COMPARISON_COL, None, number);
        COMPARE_QUANTITY: String = ("quantity", defaults::COMPARISON_QUANTITY, None, text);
        COMPARE_MAPPING: String = ("mapping", defaults::COMPARISON_MAPPING, None, text);
        COMPARE_LEVEL: u32 = ("level", "", None, number);
        COMPARE_X: usize = ("x", defaults::COMPARISON_X, None, number);
        COMPARE_Y: usize = ("y", defaults::COMPARISON_Y, None, number);
        COMPARE_OUT: String = ("out", "", Some("--out PREFIX required"), text);
        COMPARE_PORT: u16 = ("port", defaults::COMPARISON_PORT, None, number);
    }
}
