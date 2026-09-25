import { realpath } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";
import type { ExtensionAPI, ProviderConfig } from "@earendil-works/pi-coding-agent";

const models = [
  {
    id: "accounts/fireworks/models/deepseek-v4p1-flash",
    name: "Fireworks DeepSeek V4.1 Flash",
    contextWindow: 1_000_000,
    maxTokens: 384_000,
    cost: { input: 0.22, output: 0.66, cacheRead: 0.007, cacheWrite: 0 },
  },
  {
    id: "accounts/fireworks/models/glm-5p3-flash",
    name: "Fireworks GLM 5.3 Flash",
    contextWindow: 1_048_573,
    maxTokens: 131_072,
    cost: { input: 0.15, output: 0.5, cacheRead: 0.03, cacheWrite: 0 },
  },
] as const;

const modelIds = new Set<string>(models.map((model) => model.id));
const fileTools = new Set(["read", "edit", "write", "grep", "find", "ls"]);
const secretName = /^(?:\.env(?:\..*)?|\.npmrc|\.pypirc|\.netrc|credentials(?:\..*)?|auth\.json|id_rsa|id_ed25519|.*\.(?:pem|key|p12|pfx))$/i;

async function canonicalPath(path: string): Promise<string> {
  let current = path;
  const missing: string[] = [];
  for (;;) {
    try {
      return resolve(await realpath(current), ...missing.reverse());
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT" || current === dirname(current)) {
        throw error;
      }
      missing.push(basename(current));
      current = dirname(current);
    }
  }
}

function contained(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

const provider: ProviderConfig = {
  name: "Fireworks OpenAI-compatible",
  baseUrl: "https://api.fireworks.ai/inference/v1",
  apiKey: "$FIREWORKS_API_KEY",
  api: "openai-completions",
  models: models.map((model) => ({
    ...model,
    reasoning: true,
    thinkingLevelMap: {
      off: null,
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    },
    input: ["text", "image"],
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: true,
      thinkingFormat: "openai",
    },
  })),
};

export default function activate(pi: ExtensionAPI): void {
  pi.registerProvider("fireworks-uncapped", provider);
  pi.on("before_provider_request", (event) => {
    const payload = event.payload as Record<string, unknown>;
    if (typeof payload.model !== "string" || !modelIds.has(payload.model)) {
      return;
    }
    const uncapped = { ...payload };
    delete uncapped.max_tokens;
    delete uncapped.max_completion_tokens;
    return uncapped;
  });
  pi.on("tool_call", async (event, ctx) => {
    if (!fileTools.has(event.toolName)) {
      return { block: true, reason: `Disallowed tool: ${event.toolName}` };
    }
    const input = event.input as Record<string, unknown>;
    const raw = input.path ?? ".";
    if (typeof raw !== "string") {
      return { block: true, reason: "Tool path must be a string" };
    }
    for (const key of ["pattern", "glob"]) {
      const value = input[key];
      if (typeof value === "string" && (isAbsolute(value) || value.split(/[\\/]/).includes(".."))) {
        return { block: true, reason: `Disallowed ${key} path` };
      }
    }
    const root = await realpath(ctx.cwd);
    const target = await canonicalPath(resolve(root, raw));
    const rel = relative(root, target);
    if (!contained(root, target) || rel.split(sep).includes(".git") || secretName.test(basename(target))) {
      return { block: true, reason: `Disallowed worktree path: ${raw}` };
    }
  });
}
