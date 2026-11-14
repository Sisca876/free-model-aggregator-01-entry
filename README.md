# Free-model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, for the make.fun
`free-model-aggregator-01` bounty.

## How it works

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
  quality table, and the workload manifest.
- Probes every provider's catalog route in its own wire (OpenAI, Anthropic,
  Gemini) and builds one catalog, which `node main.ts --list` prints and the
  register body carries, byte for byte.
- Serves `GET /v1/models` (OpenAI shape for `Authorization: Bearer`,
  Anthropic shape for `x-api-key`), `POST /v1/chat/completions` and
  `POST /v1/messages`, each in JSON and in server-sent events.
- Routes every request to the highest-rated free model meeting its
  requirements, spreading the load where ratings tie. A provider that refuses
  (402/429/5xx) is skipped for the rest of the sweep and the request fails
  over to the next candidate; a second sweep runs when the first finds
  nothing. JSON-mode requests never ride an Anthropic-wire provider, whose
  wire cannot echo JSON mode.
- Translates requests and answers across all six client×provider wire
  combinations, carrying the system prompt, user text, images, tools, stop
  sequences, `max_tokens` and JSON mode byte for byte, so the provider's echo
  matches what the client sent.
- Polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is done.

## Determinism

Every choice derives from the brief, the requests and the providers' answers:
no wall clock, no unseeded randomness, fixed ids and timestamps in every
rendered document, sequential probes, and requests served one at a time in
arrival order. A replay leaves the same state export byte for byte.

## Layout

- `main.ts` — the server, the router, the trial loop.
- `lib/brief.ts` — the brief's documents, the catalog probe, the catalog.
- `lib/wires.ts` — wire translations, response parsing, response rendering.

## Local validation

Against the kit's sealed env and scorer (no docker), all three practice
trials score 99 with 20/20 requests valid: quality 99 (the ceiling the
providers' per-trial budgets allow), served/wire/failover/catalog 100.
Two runs of trial-0 leave byte-identical state exports.

## License

MIT — see LICENSE.
