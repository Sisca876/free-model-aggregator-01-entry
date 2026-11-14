// One server, two client wires, ten free-tier providers.
//
// The trial: read the brief, probe every provider's catalog route, build one
// catalog, listen on a port, register the port and catalog with the env,
// serve the twenty requests the env's client sends (each with the best free
// model that meets its requirements, failing over when a provider refuses),
// poll GET /work, and exit 0 when the workload is done.
//
// Routing: for each request, every free model meeting its requirements is a
// candidate, ordered by the quality table's rating (highest first), then by
// how few calls its provider has taken so far (spreading the load where
// ratings tie), then by provider and model id. The server sweeps that order
// and serves the request with the first candidate that answers; a provider
// that refuses is skipped for the rest of the sweep, and a second sweep runs
// when the first finds nothing (a refusal on a provider's failure schedule is
// a single call number: the next call may serve). Providers are never
// blacklisted across requests — a refusal says nothing about the next call.
//
// Determinism: every choice derives from the brief, the requests and the
// providers' answers. No clock, no randomness, fixed ids and timestamps in
// every rendered document, probes run one at a time in the brief's order, and
// requests are served one at a time in arrival order. A replay of the same
// trial sees the same provider answers and leaves the same state export.
//
// `node main.ts --list` prints the catalog document and exits before any
// server starts: the same object the register body carries, byte for byte.
// Outside a trial, with no brief and no env, it prints {"providers":[]}.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isFreeModel,
  isRecord,
  joinBase,
  authHeaders,
  meetsRequirements,
  probeProvider,
  ratingRowOf,
  readBrief,
  type Brief,
  type Catalog,
  type ListedModel,
  type Probe,
  type RatingRow,
  type Requirement,
  type Json,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  buildUpstreamCall,
  jsonModeOf,
  openAIError,
  openAIModelList,
  parseUpstreamAnswer,
  renderClientAnswer,
  type NormalizedAnswer,
} from './lib/wires.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** How many sweeps over the candidates one request gets before it fails. */
const MAX_SWEEPS = 2;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have been served: the manifest is in arrival order. */
let manifestIndex = 0;
/** Provider calls made so far, refusals included: every counted call spends budget. */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, one at a time in the brief's order, and build the one catalog. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.providers) {
    const probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    if (!probe.ok && probe.error !== null) log(probe.error);
    probes.push(probe);
  }
  catalog = buildCatalog(probes, brief.ratings);
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ListedModel;
  rating: number;
}

/** The free models meeting `requires`, best rating first. */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // A JSON-mode request cannot ride an Anthropic-wire provider: that wire
    // echoes json_mode false, whatever the model. OpenAI and Gemini carry it.
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row: RatingRow | null = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const aCalls = callsMade.get(a.probe.provider.id) ?? 0;
    const bCalls = callsMade.get(b.probe.provider.id) ?? 0;
    if (aCalls !== bCalls) return aCalls - bCalls;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** Requirements read off a request body, for the union with the manifest's row. */
function requirementsFromBody(clientWire: 'openai' | 'anthropic', body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: clientWire === 'openai' && jsonModeOf(body),
  };
}

/** The requirements of the request now arriving: the manifest's row, union the body's own. */
function requirementsFor(clientWire: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(clientWire, body);
  const manifestRequires = entry?.requires ?? {};
  const requires: Requirement = {
    tools: manifestRequires.tools === true || fromBody.tools === true,
    vision: manifestRequires.vision === true || fromBody.vision === true,
    json: manifestRequires.json === true || fromBody.json === true,
  };
  if (manifestRequires.min_context !== undefined) requires.min_context = manifestRequires.min_context;
  return requires;
}

// ---------------------------------------------------------------------------
// One provider call
// ---------------------------------------------------------------------------

interface AttemptOk {
  ok: true;
  answer: NormalizedAnswer;
  model: string;
}

type Attempt = AttemptOk | { ok: false };

/** Call one candidate. Any refusal, error or unparseable answer is a miss: the sweep moves on. */
async function attempt(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  anthropicVersion: string,
): Promise<Attempt> {
  const provider = candidate.probe.provider;
  const call = buildUpstreamCall(clientWire, provider.wire, body, candidate.model.id, stream);
  const url = joinBase(ENV_URL, provider.base) + call.path;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = anthropicVersion;
  if (stream) headers.accept = 'text/event-stream';
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`${provider.id}/${candidate.model.id}: unreachable: ${(err as Error).message}`);
    return { ok: false };
  }
  const text = await res.text();
  if (!res.ok) {
    log(`${provider.id}/${candidate.model.id}: HTTP ${res.status}: ${text.slice(0, 200)}`);
    return { ok: false };
  }
  const answer = parseUpstreamAnswer(provider.wire, stream, res.status, text);
  if (answer === null) {
    log(`${provider.id}/${candidate.model.id}: unparseable ${stream ? 'stream' : 'answer'}`);
    return { ok: false };
  }
  return { ok: true, answer, model: candidate.model.id };
}

/** Serve one request: sweep the candidates until one answers, then render it in the client's wire. */
async function serve(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  anthropicVersion: string,
): Promise<{ contentType: string; body: string; status: number }> {
  const requires = requirementsFor(clientWire, body);
  const jsonMode = clientWire === 'openai' && jsonModeOf(body);
  const candidates = candidatesFor(requires, jsonMode);
  for (let sweep = 0; sweep < MAX_SWEEPS; sweep += 1) {
    const refused = new Set<string>();
    for (const candidate of candidates) {
      if (refused.has(candidate.probe.provider.id)) continue;
      const result = await attempt(candidate, clientWire, body, stream, anthropicVersion);
      if (result.ok) {
        const rendered = renderClientAnswer(clientWire, stream, result.answer, result.model);
        return { contentType: rendered.contentType, body: rendered.body, status: 200 };
      }
      refused.add(candidate.probe.provider.id);
    }
  }
  // Nothing answered: a well-formed error in the client's wire, so the driver moves on.
  const errorBody =
    clientWire === 'anthropic'
      ? anthropicError('api_error', 'no free model answered this request')
      : openAIError('no free model answered this request', 'api_error');
  return { contentType: 'application/json', body: JSON.stringify(errorBody), status: 503 };
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendBody(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendError(res: ServerResponse, clientWire: 'openai' | 'anthropic', status: number, message: string): void {
  const body =
    clientWire === 'anthropic'
      ? anthropicError(status === 400 ? 'invalid_request_error' : 'api_error', message)
      : openAIError(message, status === 400 ? 'invalid_request_error' : 'api_error');
  sendBody(res, status, 'application/json', JSON.stringify(body));
}

/**
 * GET /v1/models, in the shape of the auth header it got: an x-api-key
 * request gets the Anthropic list, anything else the OpenAI one. The models
 * are exactly the catalog's free ids, derived from the catalog itself, so the
 * warm-up's two answers always match what was registered.
 */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = catalog.providers.flatMap((p) =>
    p.models.filter((m) => m.free).map((m) => ({ id: m.id, provider: p.id, display_name: m.id })),
  );
  if (typeof req.headers['x-api-key'] === 'string') {
    sendBody(res, 200, 'application/json', JSON.stringify(anthropicModelList(list)));
  } else {
    sendBody(res, 200, 'application/json', JSON.stringify(openAIModelList(list)));
  }
}

/** Requests are served one at a time, in arrival order: a promise chain is the queue. */
let chain: Promise<void> = Promise.resolve();
function enqueue<T>(work: () => Promise<T>): Promise<T> {
  const run = chain.then(work, work);
  chain = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

async function handlePost(clientWire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, clientWire, 400, 'the request body is not one JSON object');
    return;
  }
  const stream = body.stream === true;
  const version = req.headers['anthropic-version'];
  const anthropicVersion = typeof version === 'string' && version !== '' ? version : '2023-06-01';
  const result = await serve(clientWire, body, stream, anthropicVersion);
  sendBody(res, result.status, result.contentType, result.body);
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    void enqueue(() => handlePost('openai', req, res)).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, 'openai', 500, 'the server failed to answer');
    });
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    void enqueue(() => handlePost('anthropic', req, res)).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, 'anthropic', 500, 'the server failed to answer');
    });
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment it says done. A process still running at the wall scores zero. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. Wait for stdout to
    // flush so the tail is never lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    try {
      handle(req, res);
    } catch (err) {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
    }
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  const freeCount = catalog.providers.reduce((n, p) => n + p.models.filter((m) => m.free).length, 0);
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${freeCount} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
