"use strict";

// A pending Promise alone does not keep Node alive. Track only the test main.
function requireCompletion(promise) {
  let completed = false;
  const onIdle = () => {
    if (!completed) {
      console.error("FAIL: async test ended before its main promise completed");
      process.exitCode = 1;
    }
  };
  process.once("beforeExit", onIdle);
  Promise.resolve(promise).then(
    () => {
      completed = true;
      process.removeListener("beforeExit", onIdle);
    },
    (error) => {
      completed = true;
      process.removeListener("beforeExit", onIdle);
      console.error(error);
      process.exitCode = 1;
    },
  );
  return promise;
}

module.exports = { requireCompletion };
