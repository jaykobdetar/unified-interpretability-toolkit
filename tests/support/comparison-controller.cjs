"use strict";

const vm = require("node:vm"),
  { spawnSync } = require("node:child_process");

function enableModules(filename) {
  if (vm.SourceTextModule) return;
  const child = spawnSync(
    process.execPath,
    ["--experimental-vm-modules", filename],
    {
      stdio: "inherit",
    },
  );
  if (child.error) throw child.error;
  process.exit(child.status ?? 1);
}

async function load(source, context) {
  if (!/^\s*export\s/m.test(source)) {
    vm.runInContext(source.replace(/initialize\(\);\s*$/, ""), context);
    return;
  }
  const document = context.document;
  delete context.document;
  try {
    const module = new vm.SourceTextModule(source, { context });
    await module.link(() => {
      throw Error("Unexpected comparison dependency");
    });
    await module.evaluate();
    for (const name of Object.getOwnPropertyNames(module.namespace)) {
      context[name] = module.namespace[name];
    }
  } finally {
    context.document = document;
  }
}

module.exports = { enableModules, load };
