"use strict";

const vm = require("node:vm"),
  { spawnSync } = require("node:child_process"),
  fs = require("node:fs"),
  path = require("node:path"),
  { pathToFileURL } = require("node:url");

const graphs = new WeakMap(),
  boundOwners = new WeakSet();

function bindOwner(owner, context) {
  if (boundOwners.has(owner)) return;
  for (const name of [
    "AtlasTools",
    "AtlasHost",
    "AtlasProfiles",
    "atlasWorkspace",
    "atlasInferenceSelection",
    "atlasInferenceSelectionChanged",
    "atlasRefreshAvailability",
    "atlasFocusView",
    "atlasAnalyticsBridge",
  ]) {
    const holder =
      name.startsWith("Atlas") || name === "atlasWorkspace"
        ? context
        : context.window;
    if (!holder) continue;
    if (Object.hasOwn(owner, name)) holder[name] = owner[name];
    Object.defineProperty(owner, name, {
      get: () => holder[name],
      set: (value) => {
        holder[name] = value;
      },
    });
  }
  boundOwners.add(owner);
}

function enableModules(filename) {
  if (vm.SourceTextModule) return;
  const child = spawnSync(
    process.execPath,
    ["--experimental-vm-modules", filename, ...process.argv.slice(2)],
    {
      stdio: "inherit",
    },
  );
  if (child.error) throw child.error;
  process.exit(child.status ?? 1);
}

async function load(source, context, filename) {
  if (!/^\s*(?:export|import)\s/m.test(source)) {
    vm.runInContext(
      source.replace(/initialize\s*\(\s*\)\s*;\s*$/, ""),
      context,
    );
    return;
  }
  const document = context.document;
  delete context.document;
  let module;
  try {
    const graph = graphs.get(context) || new Map();
    graphs.set(context, graph);
    const create = (text, file) => {
      const key = file || Symbol();
      if (graph.has(key)) return graph.get(key);
      const item = new vm.SourceTextModule(text, {
        context,
        identifier: file,
        initializeImportMeta: (meta) => {
          if (file) meta.url = pathToFileURL(file).href;
        },
      });
      graph.set(key, item);
      return item;
    };
    module = create(source, filename && path.resolve(filename));
    if (module.status === "unlinked")
      await module.link((specifier, referencing) => {
        if (!filename || !specifier.startsWith("./"))
          throw Error("Unexpected comparison dependency");
        const file = path.resolve(
          path.dirname(referencing.identifier),
          specifier,
        );
        return create(fs.readFileSync(file, "utf8"), file);
      });
    if (module.status === "linked") await module.evaluate();
    for (const [file, item] of graph) {
      if (
        typeof file === "string" &&
        path.basename(file) === "viewer-context.js" &&
        item.status === "evaluated"
      )
        bindOwner(item.namespace.default, context);
    }
    for (const name of Object.getOwnPropertyNames(module.namespace)) {
      if (name !== "default" && name !== "module.exports")
        context[name] = module.namespace[name];
    }
    const pure = {
      "atlas-tools.js": "AtlasTools",
      "profile-client.js": "AtlasProfiles",
    }[path.basename(filename || "")];
    if (pure) context[pure] = module.namespace.default;
  } finally {
    context.document = document;
  }
  module.namespace.mount?.(context);
}

module.exports = { enableModules, load };
