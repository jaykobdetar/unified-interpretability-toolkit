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
            use super::{argument, Route};
            $(pub const $name: &str = $path;)*
            pub mod methods { $(pub const $name: &str = $method;)* }
            pub const ROUTES: &[Route] = &[
                $(Route {path:$name,method:$method,parameters:&[$(argument::$query.name),*],public:$public},)*
            ];
        }
        pub mod hosted {
            use super::argument;
            $(pub const $name: &str = $id;)*
            /// Existing private admission sets; source checks remain at the caller.
            pub fn parameters(id: &str) -> Option<&'static [&'static str]> {
                match id {
                    $($name => Some(&[$(argument::$private.name),*]),)*
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
            use super::{argument::comparison as argument, Route};
            $(pub const $name: &str = $path;)*
            pub mod methods { $(pub const $name: &str = $method;)* }
            pub const ROUTES: &[Route] = &[
                $(Route {path:$name,method:$method,parameters:&[argument::COMPARISON_IDENTITY.name,$(argument::$query.name),*],public:true},)*
            ];
        }
    };
}

comparison_routes! {
    MODEL => ("/api/comparison/model", "GET", []),
    VIEW => ("/api/comparison/view", "GET", [TENSOR, LEFT, RIGHT, MAPPING]),
    INSPECT => ("/api/comparison/inspect", "GET", [TENSOR, ROW, COL]),
    TILE => ("/api/comparison/tile", "GET", [TENSOR, QUANTITY, MAPPING, LEVEL, X, Y]),
    CALIBRATION => ("/api/comparison/calibrate", "POST", [CALIBRATION_TENSOR, ALL]),
}

/// Field parsers are lazy: source, identity, method and ownership checks stay at callers.
pub mod argument {
    use super::parameter;
    use crate::parameter::parse_indices;
    use crate::parameter::{number, text, Parameter};

    macro_rules! fields {
        ($($id:ident: $ty:ty = ($name:expr, $default:expr, $parser:expr);)*) => {
            $(pub const $id: Parameter<$ty> = Parameter::new($name, $default, None, $parser);)*
        };
    }
    fields! {
        TENSOR: usize = (parameter::TENSOR, "0", number);
        SELECTED_TENSOR: String = (parameter::TENSOR, "", text);
        SLICE: Vec<usize> = (parameter::SLICE, "", parse_indices);
        ROW: usize = (parameter::ROW, "0", number);
        COL: usize = (parameter::COL, "0", number);
        LEVEL: u32 = (parameter::LEVEL, "0", level);
        X: usize = (parameter::X, "0", number);
        Y: usize = (parameter::Y, "0", number);
        LEFT: String = (parameter::LEFT, "global_linear", text);
        RIGHT: String = (parameter::RIGHT, "global_asinh", text);
        RULE: String = (parameter::RULE, "global_linear", text);
        BINDING: String = (parameter::BINDING, "", text);
        ALL: String = (parameter::ALL, "0", text);
    }

    fn level(value: &str) -> crate::Result<u32> {
        // Preserve the original usize parse followed by a checked u32 conversion.
        Ok(number::<usize>(value)?.try_into()?)
    }

    pub mod comparison {
        use super::*;
        pub use super::{ALL, COL, LEVEL, ROW, TENSOR, X, Y};
        fields! {
            COMPARISON_IDENTITY: String = (parameter::COMPARISON_IDENTITY, "", text);
            CALIBRATION_TENSOR: usize = (parameter::TENSOR, "", number);
            LEFT: String = (parameter::LEFT, "a", text);
            RIGHT: String = (parameter::RIGHT, "b", text);
            QUANTITY: String = (parameter::QUANTITY, "delta", text);
            MAPPING: String = (parameter::MAPPING, "linear", text);
        }
    }
}
