//! Pure native route and query declarations. They do not add HTTP validation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Route {
    pub path: &'static str,
    pub method: &'static str,
    pub parameters: &'static [&'static str],
    pub public: bool,
}

pub mod parameter {
    pub const TENSOR: &str = "tensor";
    pub const SLICE: &str = "slice";
    pub const LEFT: &str = "left";
    pub const RIGHT: &str = "right";
    pub const ROW: &str = "row";
    pub const COL: &str = "col";
    pub const RULE: &str = "rule";
    pub const LEVEL: &str = "level";
    pub const X: &str = "x";
    pub const Y: &str = "y";
    pub const ALL: &str = "all";
    pub const BINDING: &str = "binding";
    pub const COMPARISON_IDENTITY: &str = "comparison_identity";
    pub const QUANTITY: &str = "quantity";
    pub const MAPPING: &str = "mapping";
}

macro_rules! viewer_routes {
    ($($name:ident => ($path:literal, $id:literal, $method:literal, $public:literal,
        [$($query:ident),*], [$($private:ident),*])),* $(,)?) => {
        pub mod viewer {
            use super::{parameter, Route};
            $(pub const $name: &str = $path;)*
            pub const ROUTES: &[Route] = &[
                $(Route {path:$name,method:$method,parameters:&[$(parameter::$query),*],public:$public},)*
            ];
        }
        pub mod hosted {
            use super::parameter;
            $(pub const $name: &str = $id;)*
            /// Existing private admission sets; source checks remain at the caller.
            pub fn parameters(id: &str) -> Option<&'static [&'static str]> {
                match id {
                    $($name => Some(&[$(parameter::$private),*]),)*
                    _ => None,
                }
            }
        }
    };
}

viewer_routes! {
    MODEL => ("/api/model", "model", "GET", true, [], []),
    PROGRESS => ("/api/progress", "progress", "GET", true, [TENSOR], [TENSOR]),
    TENSOR_STATUS => ("/api/tensor-status", "tensor-status", "GET", true, [TENSOR], [TENSOR]),
    VIEW => ("/api/view", "view", "GET", true, [TENSOR, SLICE, LEFT, RIGHT], [TENSOR, SLICE, LEFT, RIGHT]),
    INSPECT => ("/api/inspect", "inspect", "GET", true, [TENSOR, SLICE, ROW, COL, LEFT, RIGHT], [TENSOR, SLICE, ROW, COL, LEFT, RIGHT]),
    TILE => ("/tile", "tile", "GET", true, [TENSOR, SLICE, RULE, LEVEL, X, Y, BINDING], [TENSOR, SLICE, RULE, LEVEL, X, Y]),
    CALIBRATION => ("/api/calibrate", "calibration", "POST", true, [TENSOR, ALL], [TENSOR]),
    BINDING => ("/api/binding", "binding", "GET", false, [TENSOR, SLICE], [TENSOR, SLICE]),
}

macro_rules! comparison_routes {
    ($($name:ident => ($path:literal, $method:literal, [$($query:ident),*])),* $(,)?) => {
        pub mod comparison {
            use super::{parameter, Route};
            $(pub const $name: &str = $path;)*
            pub const ROUTES: &[Route] = &[
                $(Route {path:$name,method:$method,parameters:&[parameter::COMPARISON_IDENTITY,$(parameter::$query),*],public:true},)*
            ];
        }
    };
}

comparison_routes! {
    MODEL => ("/api/comparison/model", "GET", []),
    VIEW => ("/api/comparison/view", "GET", [TENSOR, LEFT, RIGHT, MAPPING]),
    INSPECT => ("/api/comparison/inspect", "GET", [TENSOR, ROW, COL]),
    TILE => ("/api/comparison/tile", "GET", [TENSOR, QUANTITY, MAPPING, LEVEL, X, Y]),
    CALIBRATION => ("/api/comparison/calibrate", "POST", [TENSOR, ALL]),
}
