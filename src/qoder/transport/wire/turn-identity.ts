/**
 * Per-turn Qoder request identity.
 *
 * A DSH turn issues one model request per reasoning step, so a single
 * subscriber prompt becomes dozens of consecutive Qoder requests. Qoder's
 * consumption records group by the request identity a conversation carries
 * (`session_id` + `chat_record_id` + model), which is why the identity of a
 * request must describe the turn, not the step: every step of one turn keeps
 * the same record id, while the next subscriber prompt opens a new one.
 *
 * The adapter never sees a turn number, and the host appends messages to a
 * turn's history while it runs — its own context, notifications, and delegated
 * agents' messages — so the turn boundary is derived from the history here. The
 * last message that is not host-appended context is the turn's anchor: the same
 * anchor keeps one record for every step, and a new anchor opens the next
 * record.
 *
 * @module dsh-provider-qoder/qoder/transport/wire/turn-identity
 */

import crypto from 'node:crypto'
import type { QoderWireMessage } from './wire-types.ts'

/**
 * Which request field carries the turn identity.
 *
 * The official client renews `request_set_id` once per agent run
 * (`AgentLifecycle.requestSetId`) while `chat_record_id` equals that request's
 * own `request_id`. Qoder's consumption panel is the only authority on which
 * field it groups by, so the transport can be switched between the candidates
 * for a live comparison before one is pinned as the default.
 */
export type QoderTurnIdentityMode = 'request-set' | 'both' | 'chat-record'

/** Bounded per-session memory: one entry per conversation this process serves. */
export interface QoderTurnTrackerOptions {
  /**
   * Maximum remembered sessions. The least recently used conversation is
   * evicted first; because an identity is rederived from the conversation, an
   * evicted session resumes the same record rather than opening a second one.
   */
  maxSessions?: number
  /** Request field the turn identity is written to. Defaults to `request-set`. */
  mode?: QoderTurnIdentityMode
}

interface SessionTurnState {
  /** Fingerprint of the input message the current record was opened by. */
  anchor: string
  /** Position of the anchor among the request's user-role messages. */
  anchorIndex: number
  /** Stable record id of that turn. */
  currentTurnId: string
  /**
   * Agent-run identity reported to Qoder as `business.id`.
   *
   * The official client creates one per agent run, so keeping it stable for
   * every request of a turn is what makes the service aggregate the turn's
   * requests into one consumption record.
   */
  currentBusinessId: string
  /** When that agent run opened, reported as `business.begin_at`. */
  turnOpenedAt: number
  /**
   * Turns opened in this session.
   *
   * The anchor alone cannot separate two consecutive turns carrying identical
   * input — a subscriber who repeats the same prompt would otherwise reuse the
   * previous record — so the ordinal keeps every new turn distinct.
   */
  turnCount: number
}

const defaultMaxSessions = 64

/**
 * Openings of everything the host appends while a turn runs.
 *
 * One subscriber prompt is what the subscriber perceives as one turn, so every
 * message the harness adds around that prompt — its own context, notifications,
 * and delegated agents' messages — must stay inside the same record. Only a
 * message outside this list is treated as a new prompt. The list is
 * evidence-based: every entry was observed in live DSH transcripts, and an
 * offline audit replays real sessions to keep it aligned. An unknown future
 * wrapper therefore opens one extra record instead of merging two turns.
 */
const injectedContextOpenings: readonly string[] = [
  '<system-reminder>',
  '<openviking-context',
  '<goal_complete>',
  'Current runtime context. This snapshot supersedes earlier runtime-context',
  'This is an automatically generated checkpoint condensing',
  '[model changed:',
  'Background subagent ',
  'background job ',
  'The approval policy changed',
  'Agent ',
]

/** Wire text of one user message, used for injection classification. */
function messageText(message: QoderWireMessage): string {
  const content = message.content
  if (typeof content === 'string') return content
  if (content === null) return ''
  return content.map(part => part.type === 'text' ? part.text : '').join('')
}

/**
 * Whether one user-role message is host context rather than subscriber input.
 *
 * Exported for the offline audit that checks the opening list against real
 * transcripts, so the list cannot silently drift from what the host appends.
 */
export function isInjectedQoderContext(message: QoderWireMessage): boolean {
  const text = messageText(message).trimStart()
  return injectedContextOpenings.some(opening => text.startsWith(opening))
}

/** Stable record id of one session turn. */
function turnIdFor(sessionId: string, turnCount: number, anchor: string): string {
  return crypto.createHash('sha256')
    .update('qoder-turn')
    .update('\0')
    .update(sessionId)
    .update('\0')
    .update(String(turnCount))
    .update('\0')
    .update(anchor)
    .digest('hex')
    .slice(0, 16)
}

/**
 * Stable fingerprint of one message.
 *
 * The anchor is content, not position: the host keeps appending history while a
 * turn runs, so a positional anchor would drift within a turn and split it.
 * The anchor is also not hashed alone by role, so an identical prompt in a later
 * turn cannot silently reuse the previous turn's record.
 */
function messageFingerprint(message: QoderWireMessage): string {
  const hash = crypto.createHash('sha256')
  hash.update(message.role)
  hash.update('\0')
  const content = message.content
  if (typeof content === 'string') {
    hash.update(content)
  } else if (content !== null) {
    for (const part of content) {
      hash.update('\0')
      hash.update(part.type)
      hash.update('\0')
      hash.update(part.type === 'text' ? part.text : part.image_url.url)
    }
  }
  return hash.digest('hex').slice(0, 16)
}

/**
 * Resolves the record identity of the turn one request belongs to.
 *
 * One tracker instance owns its sessions, so tests and transports stay
 * independent. State is created lazily per session and bounded by
 * `maxSessions`.
 */
export class QoderTurnTracker {
  private readonly maxSessions: number
  /** Request field the resolved turn id is written to. */
  readonly mode: QoderTurnIdentityMode
  private readonly sessions = new Map<string, SessionTurnState>()

  constructor(options: QoderTurnTrackerOptions = {}) {
    this.maxSessions = options.maxSessions ?? defaultMaxSessions
    this.mode = options.mode ?? 'request-set'
  }

  /**
   * Record id shared by every step of `sessionId`'s current turn.
   *
   * `null` means the request carries no session identity, so no conversation
   * identity can be derived and the caller must fall back to a per-request id.
   */
  resolveTurnRecordId(sessionId: string | undefined, messages: readonly QoderWireMessage[]): string | null {
    if (sessionId === undefined || sessionId.length === 0) return null

    let state = this.sessions.get(sessionId)
    if (state === undefined) {
      state = { anchor: '', anchorIndex: -1, currentTurnId: '', currentBusinessId: '', turnOpenedAt: 0, turnCount: 0 }
      this.claimSession(sessionId, state)
    } else {
      // Refresh recency so eviction drops the coldest conversation.
      this.sessions.delete(sessionId)
      this.sessions.set(sessionId, state)
    }

    // Host-appended context belongs to the open record and can never anchor a
    // new one. The anchor is the last message carrying real input; its position
    // is what separates a repeated prompt from the open record, because a turn's
    // own later steps only ever append after the anchor, while a genuinely new
    // prompt appears after everything the open record already contained.
    let anchor: string | undefined
    let anchorIndex = -1
    let userIndex = 0
    for (const message of messages) {
      if (message.role !== 'user') continue
      const index = userIndex++
      if (isInjectedQoderContext(message)) continue
      anchor = messageFingerprint(message)
      anchorIndex = index
    }

    // A different anchor text is a new prompt. Identical text at a later
    // position is also a new prompt — a subscriber who repeats the same words —
    // and the ordinal keeps that record distinct from the previous one.
    const openedTurn = anchor !== undefined
      && (anchor !== state.anchor || anchorIndex > state.anchorIndex)
    if (openedTurn && anchor !== undefined) {
      state.anchor = anchor
      state.anchorIndex = anchorIndex
      state.turnCount += 1
      state.currentTurnId = turnIdFor(sessionId, state.turnCount, anchor)
      state.currentBusinessId = crypto.randomUUID()
      state.turnOpenedAt = Date.now()
    } else if (state.currentTurnId === '') {
      // A request without subscriber input still belongs to a record; it stays
      // that record until the first real prompt arrives.
      state.turnCount += 1
      state.currentTurnId = turnIdFor(sessionId, state.turnCount, '')
      state.currentBusinessId = crypto.randomUUID()
      state.turnOpenedAt = Date.now()
    }

    return state.currentTurnId
  }

  /**
   * Agent-run identity the Qoder service reports consumption against.
   *
   * The official client stamps one `business.id` per agent run and keeps it for
   * every request that run makes, which is what makes one subscriber turn one
   * consumption record. A request outside any conversation still gets its own.
   */
  resolveBusinessId(
    sessionId: string | undefined,
    messages: readonly QoderWireMessage[],
  ): { businessId: string; beginAt: number } {
    if (sessionId !== undefined && sessionId.length > 0) {
      this.resolveTurnRecordId(sessionId, messages)
      const state = this.sessions.get(sessionId)
      if (state !== undefined && state.currentBusinessId !== '') {
        return { businessId: state.currentBusinessId, beginAt: state.turnOpenedAt }
      }
    }
    return { businessId: crypto.randomUUID(), beginAt: Date.now() }
  }

  /** Forget one session, e.g. when its transport is disposed. */
  forget(sessionId: string): void {
    this.sessions.delete(sessionId)
  }

  /** Forget every session. */
  clear(): void {
    this.sessions.clear()
  }

  private claimSession(sessionId: string, state: SessionTurnState): void {
    this.sessions.set(sessionId, state)
    // Evict the coldest conversations, keeping the identity being created.
    while (this.sessions.size > this.maxSessions) {
      const oldest = this.sessions.keys().next()
      if (oldest.done === true) return
      this.sessions.delete(oldest.value)
    }
    // Re-insert so the newest conversation is also the most recently used.
    this.sessions.delete(sessionId)
    this.sessions.set(sessionId, state)
  }
}
