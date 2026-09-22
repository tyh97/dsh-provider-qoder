/**
 * Per-turn Qoder request identity.
 *
 * Qoder groups consumption records by the identity a conversation request
 * carries, so one subscriber prompt must resolve to one record no matter how
 * many reasoning steps it runs, and the next prompt must open a new one.
 */

import assert from 'node:assert/strict'
import test from 'node:test'
import { createUserMessage, type GenerateOptions } from '@deepseek-ai/dsh-llm'
import { buildQoderRequestBody } from '../src/qoder/transport/wire/serialize.ts'
import type { QoderWireMessage } from '../src/qoder/transport/wire/wire-types.ts'
import { QoderTurnTracker } from '../src/qoder/transport/wire/turn-identity.ts'

function user(text: string) {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

/** Wire-level user message, the shape the turn tracker inspects. */
function wireUser(text: string): QoderWireMessage {
  return { role: 'user', content: [{ type: 'text', text }] }
}

/** Wire-level assistant turn carrying a tool call. */
function wireToolCall(id: string): QoderWireMessage {
  return {
    role: 'assistant',
    content: null,
    tool_calls: [{ id, type: 'function', function: { name: 'read', arguments: '{}' } }],
  }
}

/** Wire-level tool result. */
function wireToolResult(id: string): QoderWireMessage {
  return { role: 'tool', content: [{ type: 'text', text: 'file body' }], tool_call_id: id }
}

/** Host-appended context, as observed in live DSH transcripts. */
function injected(text: string): QoderWireMessage {
  return wireUser(`<system-reminder>${text}</system-reminder>`)
}

function options(messages: GenerateOptions['messages'], sessionId?: string): GenerateOptions {
  return {
    provider: 'dsh-provider-qoder',
    model: 'cmodel',
    messages,
    ...sessionId === undefined ? {} : { sessionId },
  } as GenerateOptions
}

test('every step of one turn shares a single consumption record', () => {
  const tracker = new QoderTurnTracker()
  const session = 'identity-session-1'
  const prompt = wireUser('Fix the failing test')

  const first = tracker.resolveTurnRecordId(session, [prompt])
  const second = tracker.resolveTurnRecordId(session, [
    prompt,
    wireToolCall('call-1'),
    wireToolResult('call-1'),
  ])
  const third = tracker.resolveTurnRecordId(session, [
    prompt,
    wireToolCall('call-2'),
    injected('memory'),
    wireUser('<openviking-context>Relevant memory</openviking-context>'),
  ])

  assert.equal(typeof first, 'string')
  assert.equal(second, first)
  assert.equal(third, first)
})

test('the next subscriber prompt opens a new record', () => {
  const tracker = new QoderTurnTracker()
  const session = 'identity-session-2'
  const firstPrompt = wireUser('First request')
  const secondPrompt = wireUser('Second request')

  const first = tracker.resolveTurnRecordId(session, [firstPrompt])
  const second = tracker.resolveTurnRecordId(session, [firstPrompt, secondPrompt])

  assert.notEqual(second, first)
  // The new turn stays stable for the rest of its own steps.
  assert.equal(tracker.resolveTurnRecordId(session, [firstPrompt, secondPrompt, injected('more context')]), second)
})

test('mid-turn messages that are not a new prompt stay in the open record', () => {
  const tracker = new QoderTurnTracker()
  const session = 'identity-session-3'
  const prompt = wireUser('Review this change')
  const parentMessage = wireUser('Agent 123 sent a message: also check the retry path')

  const first = tracker.resolveTurnRecordId(session, [prompt])
  const second = tracker.resolveTurnRecordId(session, [prompt, injected('memory')])
  const third = tracker.resolveTurnRecordId(session, [prompt, parentMessage])

  assert.equal(second, first)
  // A delegated agent's message belongs to this run, not to a second prompt.
  assert.equal(third, first)
})

test('a conversation without host context still splits per prompt', () => {
  const tracker = new QoderTurnTracker()
  const session = 'identity-session-5'
  const firstPrompt = wireUser('First')
  const secondPrompt = wireUser('Second')
  const thirdPrompt = wireUser('Third')

  const first = tracker.resolveTurnRecordId(session, [firstPrompt])
  const second = tracker.resolveTurnRecordId(session, [firstPrompt, secondPrompt])
  const third = tracker.resolveTurnRecordId(session, [firstPrompt, secondPrompt, thirdPrompt])

  assert.notEqual(second, first)
  assert.notEqual(third, second)
  // Re-resolving the same conversation is stable.
  assert.equal(tracker.resolveTurnRecordId(session, [firstPrompt, secondPrompt, thirdPrompt]), third)
})

test('host-injected context of every known shape stays inside the turn', () => {
  const tracker = new QoderTurnTracker()
  const session = 'identity-session-4'
  const prompt = wireUser('Audit the transport')
  const injections = [
    wireUser('<system-reminder>Workspace instructions that are relevant</system-reminder>'),
    wireUser('<openviking-context>Relevant memory from OpenViking</openviking-context>'),
    wireUser('Current runtime context. This snapshot supersedes earlier runtime-context snapshots.'),
    wireUser('[model changed: provider/model]'),
    wireUser('Background subagent abc finished'),
    wireUser('background job pwsh-1 completed'),
    wireUser('This is an automatically generated checkpoint condensing the session'),
    wireUser('<goal_complete>Objective reached</goal_complete>'),
  ]

  const first = tracker.resolveTurnRecordId(session, [prompt])
  assert.equal(tracker.resolveTurnRecordId(session, [prompt, ...injections]), first)
  assert.equal(
    tracker.resolveTurnRecordId(session, [prompt, ...injections, wireUser('<system-reminder>and more</system-reminder>')]),
    first,
  )
})

test('sessions never share a record id', () => {
  const tracker = new QoderTurnTracker()

  const one = tracker.resolveTurnRecordId('identity-session-a', [wireUser('Same text')])
  const two = tracker.resolveTurnRecordId('identity-session-b', [wireUser('Same text')])

  assert.notEqual(one, two)
})

test('requests without a session identity stay independent records', () => {
  const tracker = new QoderTurnTracker()

  assert.equal(tracker.resolveTurnRecordId(undefined, [wireUser('Ping')]), null)
  assert.equal(tracker.resolveTurnRecordId('', [wireUser('Ping')]), null)
})

test('the request envelope carries the official client identity model', async () => {
  const prompt = user('Continue')
  const first = await buildQoderRequestBody(options([prompt], 'identity-session-envelope'), 'user-42')
  const second = await buildQoderRequestBody(options([prompt], 'identity-session-envelope'), 'user-42')
  const nextTurn = await buildQoderRequestBody(options([prompt, user('Next')], 'identity-session-envelope'), 'user-42')

  // The official client renews request_set_id once per agent run and sets
  // chat_record_id to the request's own id, so one run keeps one set id while
  // every request carries a fresh record id.
  assert.equal(first.request_set_id, second.request_set_id)
  assert.notEqual(first.chat_record_id, second.chat_record_id)
  assert.notEqual(first.request_id, second.request_id)
  assert.notEqual(nextTurn.request_set_id, first.request_set_id)
  // The conversation key stays untouched.
  assert.match(first.session_id, /^[a-f0-9]{16}-identity-session-envelope$/)
})

test('one agent run reports one business id for every request', async () => {
  const prompt = user('Continue')
  const first = await buildQoderRequestBody(options([prompt], 'identity-session-business'), 'user-42')
  const second = await buildQoderRequestBody(options([prompt], 'identity-session-business'), 'user-42')
  const nextTurn = await buildQoderRequestBody(
    options([prompt, user('Next')], 'identity-session-business'),
    'user-42',
  )

  // The service aggregates consumption by the agent run, so a turn's requests
  // must share one business id while the next prompt opens a new run.
  assert.equal(first.business.id, second.business.id)
  assert.notEqual(nextTurn.business.id, first.business.id)
  // The run keeps its own start time instead of one per request.
  assert.equal(first.business.begin_at, second.business.begin_at)
  assert.ok(nextTurn.business.begin_at >= first.business.begin_at)
})

test('the tracker mode selects which field carries the turn id', async () => {
  const prompt = user('Continue')
  const optionsFor = (messages: GenerateOptions['messages']) => options(messages, 'identity-session-mode')

  assert.equal(new QoderTurnTracker().mode, 'request-set')
  assert.equal(new QoderTurnTracker({ mode: 'both' }).mode, 'both')
  assert.equal(new QoderTurnTracker({ mode: 'chat-record' }).mode, 'chat-record')

  const first = await buildQoderRequestBody(optionsFor([prompt]), 'user-42')
  const second = await buildQoderRequestBody(optionsFor([prompt]), 'user-42')
  assert.equal(first.request_set_id, second.request_set_id)
  assert.notEqual(first.chat_record_id, second.chat_record_id)
})

test('repeating the same prompt still opens a distinct record per turn', () => {
  const tracker = new QoderTurnTracker()
  const session = 'identity-session-repeat'
  const prompt = wireUser('继续')

  const first = tracker.resolveTurnRecordId(session, [prompt])
  // The subscriber repeats the same text as the next prompt: same anchor, but a
  // different turn, so it must not reuse the previous record.
  const second = tracker.resolveTurnRecordId(session, [prompt, injected('memory between turns'), prompt])

  assert.notEqual(second, first)
  // The repeated turn is still stable for its own later steps.
  assert.equal(
    tracker.resolveTurnRecordId(session, [prompt, injected('memory between turns'), prompt, injected('more')]),
    second,
  )
})

test('the tracker keeps a bounded, deterministic per-session identity', () => {
  const tracker = new QoderTurnTracker({ maxSessions: 2 })

  const kept = tracker.resolveTurnRecordId('identity-keep', [wireUser('keep')])
  const other = tracker.resolveTurnRecordId('identity-other', [wireUser('other')])
  // Touch the first session so the second becomes the coldest.
  assert.equal(tracker.resolveTurnRecordId('identity-keep', [wireUser('keep')]), kept)
  tracker.resolveTurnRecordId('identity-third', [wireUser('third')])

  // The most recently used conversation keeps its record.
  assert.equal(tracker.resolveTurnRecordId('identity-keep', [wireUser('keep')]), kept)
  // A conversation whose state was evicted rederives the same record from the
  // conversation itself, so eviction can never split one turn into two records.
  assert.equal(tracker.resolveTurnRecordId('identity-other', [wireUser('other')]), other)
})
