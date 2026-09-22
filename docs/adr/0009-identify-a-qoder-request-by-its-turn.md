# Identify a Qoder request by its turn, not by its step

A DSH turn issues one model request per reasoning step, so a single subscriber prompt becomes dozens of consecutive Qoder requests. Qoder's Credits panel lists one row per agent run and totals that run's requests, while the transport reported a fresh `business.id` on every request, so one subscriber prompt appeared as dozens of disjoint sub-minute entries instead of one record covering the turn.

The transport now reports an agent run the way the official client does, which is what makes the service aggregate a turn:

- `business.id` and `business.begin_at` identify one agent run and are reused by every request of that turn. This is the aggregation key: the official CLI creates one `business.id` per agent run (`id: L || randomUUID()`), so a stable value here is what turns a turn's requests into one consumption record.
- `request_set_id` carries the same turn identity, matching the official client's per-run `AgentLifecycle.requestSetId`.
- `chat_record_id` equals that request's own `request_id`, and `request_id` is a fresh UUID per request, matching the official client exactly.
- `session_id` keeps identifying the conversation and is unchanged.

The turn boundary is derived in `qoder/transport/wire/turn-identity.ts`, because the adapter sees no turn number. The anchor is the last user-role message that is not host-appended context, identified by content *and* position among the request's user-role messages. Position is what separates a repeated prompt from the open record: a turn's own later steps only ever append after its anchor, while a genuinely new prompt appears at a position the open record never carried. Messages the host appends while a turn runs — its own context blocks, background notifications, checkpoint and goal markers, and delegated agents' messages — are folded into the open record. The opening list is evidence-based: every entry was observed in live DSH transcripts, and an offline audit replays real sessions to confirm that the classifier folds every host-appended message and that each turn resolves to exactly one identity.

Three candidate keys were tested against the live panel before this one. A stable `session_id`, a turn-stable `chat_record_id`, and a turn-stable `request_set_id` each left the panel showing one row per request; only the agent-run `business.id` merged them. The identity model therefore follows the official client field by field instead of deriving a single synthetic record id.

Two consequences follow. An unknown future wrapper opens one extra record instead of merging two turns, because a missing entry can only over-split. And a process that restarts mid-turn cannot recover the open run, so the turn that was running is split into two; the tracker keeps identities bounded per session (least recently used, 64 conversations) and rederives the same identity for a conversation whose state was evicted, so eviction itself never splits a turn.
