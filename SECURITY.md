# Security

Network Evolution Lab is a single-user research tool. Keep the native panel and the Compose web proxy on localhost. The panel validates Host and Origin and requires a per-process token for start and stop requests. Grant filesystem access to trusted users.

The structured evaluator accepts an allowlisted feature schema and bounded numeric conditions, weights, depth and node count. Legacy and exported Python detectors are parsed and evaluated with a restricted numeric AST interpreter. Use `evaluation.classify` to load them. Saved Python candidates and reports should be treated as untrusted when imported from other environments.

Store the Groq key in your local environment or `.env`. The metrics container has no provider credentials and sees a read-only copy of the official train CSV. Official test data stays in the worker dataset mount. Model prompts in Hybrid contain rules, allowed features and aggregate statistics. Usage artifacts contain request counts, status and reported tokens. The Compose administrator can inspect the worker environment.

The optional Shinka process and its dependencies have a broader execution surface than the structured engine. Run it with isolated files and a dedicated Groq credential. Its local gateway checks a per-run token and enforces an explicit attempt budget. The integration is experimental.

Report security issues privately to the repository maintainers. Include the affected version, impact and minimal reproduction. Remove credentials, raw events and identifying data from reports.

The panel serves archived HTML with a Content Security Policy that disables scripts and external resources. Review locally opened copies of imported reports separately. Before a public release, audit tracked files, staged files, commit metadata and screenshots. Review publication consent for any personal commit metadata.
