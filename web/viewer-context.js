// The page owns one shared lifecycle context. Imports do not mount UI or start work.
const context = {};

// Preserve the existing optional host binding for synchronous Node consumers.
if (
  typeof process !== "undefined" &&
  typeof process.getBuiltinModule === "function"
) {
  Object.defineProperty(context, "AtlasHost", {
    get: () => globalThis.AtlasHost,
    set: (value) => {
      globalThis.AtlasHost = value;
    },
  });
}
export default context;
