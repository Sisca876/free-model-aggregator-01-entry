// The three wires, and every translation between them.
//
// The client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic
// or Gemini. A request is translated into the provider's wire, the provider's
// answer is parsed into one normalized shape, and that shape is rendered in
// the client's wire, in JSON or in server-sent events.
//
// The one rule that shapes every translation: the provider echoes what it
// parsed — the system text, the user text, the tool names, max_tokens, the
// stop sequences, the image bytes and JSON mode — and the client compares
// that echo with what it sent. A field rewritten or dropped loses the
// request. So every translation carries those fields byte for byte: the same
// text, the same base64 data, the same names in the same order, a number
// max_tokens only when the request carried one.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness, fixed ids and timestamps in every rendered document.

import { isRecord, type Json } from './brief.ts';

/** One provider answer, normalized: the text whole, the usage, the stop family. */
export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** 0 stop, 1 length, 2 tool calls, 3 refusal — the checker's stop families. */
  stopFamily: 0 | 1 | 2 | 3;
}

/** An upstream call's target: the path under the provider's base and the body. */
export interface UpstreamCall {
  path: string;
  body: Json;
}

/** A fixed epoch second for every rendered completion's `created`. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's Anthropic `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How wide one streamed text delta is, in characters. */
const STREAM_PIECE = 13;

/** The text of a string or of a block array's text blocks, joined with newlines. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The text of a Gemini parts array: its text parts, joined with newlines. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('\n');
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The stop family of a wire's stop reason; unknown reasons read as a plain stop. */
export function stopFamilyOf(reason: unknown): 0 | 1 | 2 | 3 {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

const OPENAI_STOP = ['stop', 'length', 'tool_calls', 'content_filter'] as const;
const ANTHROPIC_STOP = ['end_turn', 'max_tokens', 'tool_use', 'refusal'] as const;

/** Split text into streamed pieces; an empty text is one empty piece. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event frame: the event line when named, the data line, the blank line. */
export function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE frame as parsed: its event name (or null) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse buffered SSE text into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const chunk of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Images: the same bytes in every wire's spelling, so the echo's hash matches.
// ---------------------------------------------------------------------------

/** A `data:<media>;base64,<data>` URL split, or null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const match = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** An OpenAI image_url part as an Anthropic image block. */
function openAIImageToAnthropic(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/** An Anthropic image block as an OpenAI image_url part. */
function anthropicImageToOpenAI(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** An OpenAI image_url part as a Gemini part (inline data, or a file reference). */
function openAIImageToGemini(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini part. */
function anthropicImageToGemini(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { inlineData: { mimeType: media, data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

// ---------------------------------------------------------------------------
// Tools: the same names, in the same order, in every wire's spelling.
// ---------------------------------------------------------------------------

/** OpenAI tool definitions as Anthropic tool definitions. */
function openAIToolsToAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const out: Json[] = [];
  for (const tool of tools) {
    if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
    out.push({
      name: tool.function.name,
      description: typeof tool.function.description === 'string' ? tool.function.description : '',
      input_schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
    });
  }
  return out;
}

/** Anthropic tool definitions as OpenAI tool definitions. */
function anthropicToolsToOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  const out: Json[] = [];
  for (const tool of tools) {
    if (!isRecord(tool) || typeof tool.name !== 'string') continue;
    out.push({
      type: 'function',
      function: {
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      },
    });
  }
  return out;
}

/** Tool definitions of either client wire as one Gemini tools block. */
function toolsToGemini(openAITools: unknown, anthropicTools: unknown): Json[] {
  const declarations: Json[] = [];
  if (Array.isArray(openAITools)) {
    for (const tool of openAITools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      declarations.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  if (Array.isArray(anthropicTools)) {
    for (const tool of anthropicTools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      declarations.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return declarations.length === 0 ? [] : [{ functionDeclarations: declarations }];
}

/** A tool result's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Request builders: the client's request in the provider's wire.
// ---------------------------------------------------------------------------

/** The request's max_tokens: OpenAI's max_tokens or max_completion_tokens, Anthropic's max_tokens. */
function maxTokensOf(body: Json): number | null {
  const direct = body.max_tokens;
  if (typeof direct === 'number' && Number.isFinite(direct)) return direct;
  const completion = body.max_completion_tokens;
  if (typeof completion === 'number' && Number.isFinite(completion)) return completion;
  return null;
}

/** Whether an OpenAI request asks for JSON mode. */
export function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The system text of an OpenAI request: its system messages' text, joined with newlines. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = blocksText(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** An OpenAI request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = openAISystemText(body);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = blocksText(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsObject(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : blocksText(raw.content),
          },
        ],
      });
      continue;
    }
    // A user message: text and image blocks carried over, in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url)) {
        const image = openAIImageToAnthropic(block.image_url);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic request as an OpenAI Chat Completions request naming `model`. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = blocksText(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else if (block.type === 'image') {
        const part = anthropicImageToOpenAI(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  const tools = anthropicToolsToOpenAI(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI message as Gemini contents. Tool messages ride as `model` contents. */
function openAIMessageToGemini(message: Json): Json[] {
  if (message.role === 'system') return [];
  if (message.role === 'tool') {
    // A tool message carries no user text: as a `model` content the provider's
    // parser skips it, so the user text it echoes stays the user's own.
    return [
      {
        role: 'model',
        parts: [
          {
            functionResponse: {
              name: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
              response: { result: typeof message.content === 'string' ? message.content : blocksText(message.content) },
            },
          },
        ],
      },
    ];
  }
  if (message.role === 'assistant') {
    const parts: Json[] = [];
    const text = blocksText(message.content);
    if (text !== '') parts.push({ text });
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({
        functionCall: {
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argumentsObject(call.function.arguments),
        },
      });
    }
    return parts.length === 0 ? [] : [{ role: 'model', parts }];
  }
  // A user message: text and image parts, in order.
  const parts: Json[] = [];
  if (typeof message.content === 'string') {
    if (message.content !== '') parts.push({ text: message.content });
  } else if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url)) {
        const part = openAIImageToGemini(block.image_url);
        if (part !== null) parts.push(part);
      }
    }
  }
  return parts.length === 0 ? [] : [{ role: 'user', parts }];
}

/** One Anthropic message as Gemini contents. */
function anthropicMessageToGemini(message: Json): Json[] {
  const assistant = message.role === 'assistant';
  const parts: Json[] = [];
  if (typeof message.content === 'string') {
    if (message.content !== '') parts.push({ text: message.content });
  } else if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image') {
        const part = anthropicImageToGemini(block);
        if (part !== null) parts.push(part);
      } else if (block.type === 'tool_use') {
        parts.push({
          functionCall: {
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          },
        });
      } else if (block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            response: { result: toolResultText(block.content) },
          },
        });
      }
    }
  }
  if (parts.length === 0) return [];
  return [{ role: assistant ? 'model' : 'user', parts }];
}

/** An OpenAI or Anthropic request as a Gemini generateContent body. */
export function toGemini(body: Json, clientWire: 'openai' | 'anthropic', model: string, stream: boolean): UpstreamCall {
  const contents: Json[] = [];
  const systemText = clientWire === 'openai' ? openAISystemText(body) : blocksText(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const converted = clientWire === 'openai' ? openAIMessageToGemini(raw) : anthropicMessageToGemini(raw);
    contents.push(...converted);
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = toolsToGemini(clientWire === 'openai' ? body.tools : undefined, clientWire === 'anthropic' ? body.tools : undefined);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = clientWire === 'openai' ? stringList(body.stop) : stringList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (clientWire === 'openai' && jsonModeOf(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  const path = `/v1beta/models/${encodeURIComponent(model)}:${verb}${stream ? '?alt=sse' : ''}`;
  return { path, body: out };
}

/**
 * Build one provider call: the path under the provider's base and the body in
 * the provider's wire. Same-wire calls pass the client's body through with
 * the routed model named; cross-wire calls are translated.
 */
export function buildUpstreamCall(
  clientWire: 'openai' | 'anthropic',
  providerWire: 'openai' | 'anthropic' | 'gemini',
  body: Json,
  model: string,
  stream: boolean,
): UpstreamCall {
  if (providerWire === 'gemini') return toGemini(body, clientWire, model, stream);
  if (providerWire === 'openai') {
    if (clientWire === 'openai') {
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return { path: '/v1/chat/completions', body: out };
    }
    return { path: '/v1/chat/completions', body: anthropicToOpenAI(body, model) };
  }
  if (clientWire === 'anthropic') {
    return { path: '/v1/messages', body: { ...body, model } };
  }
  return { path: '/v1/messages', body: openAIToAnthropic(body, model) };
}

// ---------------------------------------------------------------------------
// Response parsers: a provider's JSON or SSE answer, normalized.
// ---------------------------------------------------------------------------

/** The text of an OpenAI message: its string content, or its text blocks joined. */
function openAIMessageText(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  if (!Array.isArray(message.content)) return '';
  const parts: string[] = [];
  for (const block of message.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/** The text of an Anthropic message: its text blocks joined. */
function anthropicMessageText(body: Json): string {
  if (!Array.isArray(body.content)) return '';
  const parts: string[] = [];
  for (const block of body.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/** The text of a Gemini candidate: its parts' text joined. */
function geminiCandidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  const parts: string[] = [];
  for (const part of content.parts) {
    if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('');
}

/** Parse an OpenAI JSON completion. */
function parseOpenAIJson(body: Json): NormalizedAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: openAIMessageText(message),
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
  };
}

/** Parse an OpenAI SSE stream, buffered whole. */
function parseOpenAIStream(text: string): NormalizedAnswer | null {
  let content = '';
  let input = 0;
  let output = 0;
  let stop: unknown = null;
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') content += delta.content;
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) stop = choice.finish_reason;
  }
  if (!sawAny) return null;
  return { text: content, inputTokens: input, outputTokens: output, stopFamily: stopFamilyOf(stop) };
}

/** Parse an Anthropic JSON message. */
function parseAnthropicJson(body: Json): NormalizedAnswer | null {
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: anthropicMessageText(body),
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
  };
}

/** Parse an Anthropic SSE stream, buffered whole. */
function parseAnthropicStream(text: string): NormalizedAnswer | null {
  let content = '';
  let input = 0;
  let output = 0;
  let stop: unknown = null;
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      content += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  if (!sawAny) return null;
  return { text: content, inputTokens: input, outputTokens: output, stopFamily: stopFamilyOf(stop) };
}

/** Parse a Gemini JSON generateContent answer. */
function parseGeminiJson(body: Json): NormalizedAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: geminiCandidateText(candidate),
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

/** Parse a Gemini SSE stream, buffered whole. */
function parseGeminiStream(text: string): NormalizedAnswer | null {
  let content = '';
  let input = 0;
  let output = 0;
  let stop: unknown = null;
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    sawAny = true;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      content += geminiCandidateText(candidate);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(chunk.usageMetadata)) {
      input = numOrZero(chunk.usageMetadata.promptTokenCount);
      output = numOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text: content, inputTokens: input, outputTokens: output, stopFamily: stopFamilyOf(stop) };
}

/** Parse one provider answer — JSON or SSE, by its wire — into the normalized shape. */
export function parseUpstreamAnswer(
  providerWire: 'openai' | 'anthropic' | 'gemini',
  stream: boolean,
  status: number,
  text: string,
): NormalizedAnswer | null {
  if (status < 200 || status >= 300) return null;
  if (!stream) {
    const body = jsonOrNull(text);
    if (body === null) return null;
    if (providerWire === 'openai') return parseOpenAIJson(body);
    if (providerWire === 'anthropic') return parseAnthropicJson(body);
    return parseGeminiJson(body);
  }
  if (providerWire === 'openai') return parseOpenAIStream(text);
  if (providerWire === 'anthropic') return parseAnthropicStream(text);
  return parseGeminiStream(text);
}

// ---------------------------------------------------------------------------
// Renderers: the normalized answer in the client's wire, JSON or SSE.
// ---------------------------------------------------------------------------

/** An OpenAI JSON completion carrying the normalized answer. */
export function renderOpenAIJson(answer: NormalizedAnswer, model: string): string {
  return JSON.stringify({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: OPENAI_STOP[answer.stopFamily],
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
}

/** An OpenAI SSE stream carrying the normalized answer, ending with [DONE]. */
export function renderOpenAIStream(answer: NormalizedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, OPENAI_STOP[answer.stopFamily])));
  frames.push(
    sseFrame(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Anthropic JSON message carrying the normalized answer. */
export function renderAnthropicJson(answer: NormalizedAnswer, model: string): string {
  return JSON.stringify({
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: ANTHROPIC_STOP[answer.stopFamily],
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

/** An Anthropic SSE stream carrying the normalized answer, in event order. */
export function renderAnthropicStream(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }),
    );
  }
  frames.push(
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: ANTHROPIC_STOP[answer.stopFamily], stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** Render the normalized answer for the client: the wire and shape it asked for. */
export function renderClientAnswer(
  clientWire: 'openai' | 'anthropic',
  stream: boolean,
  answer: NormalizedAnswer,
  model: string,
): { contentType: string; body: string } {
  if (clientWire === 'openai') {
    return stream
      ? { contentType: 'text/event-stream', body: renderOpenAIStream(answer, model) }
      : { contentType: 'application/json', body: renderOpenAIJson(answer, model) };
  }
  return stream
    ? { contentType: 'text/event-stream', body: renderAnthropicStream(answer, model) }
    : { contentType: 'application/json', body: renderAnthropicJson(answer, model) };
}

// ---------------------------------------------------------------------------
// Model lists and errors, in each wire's shape.
// ---------------------------------------------------------------------------

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models for an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models for an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
