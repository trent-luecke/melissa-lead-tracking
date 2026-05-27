// cloudflare/worker.js
// Slack Events API webhook receiver for Melissa Lead Tracking.
// Verifies Slack signatures, filters qualifying DM thread replies,
// and triggers the reply.yml GitHub Actions workflow via workflow_dispatch.

async function verifySlackSignature(signingSecret, timestamp, body, signature) {
  if (!signingSecret || !timestamp || !body || !signature) return false;

  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(signingSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const baseString = `v0:${timestamp}:${body}`;
  const rawSignature = await crypto.subtle.sign("HMAC", key, encoder.encode(baseString));
  const computedSig =
    "v0=" +
    Array.from(new Uint8Array(rawSignature))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

  // Constant-time comparison to prevent timing attacks
  if (computedSig.length !== signature.length) return false;
  let mismatch = 0;
  for (let i = 0; i < computedSig.length; i++) {
    mismatch |= computedSig.charCodeAt(i) ^ signature.charCodeAt(i);
  }
  return mismatch === 0;
}

async function dispatchToGitHub(env, threadTs, replyText) {
  const resp = await fetch(
    `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/reply.yml/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_PAT}`,
        Accept: "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "melissa-lead-manager",
      },
      body: JSON.stringify({
        ref: "main",
        inputs: {
          thread_ts: threadTs,
          reply_text: replyText,
        },
      }),
    }
  );

  if (!resp.ok) {
    console.error(`GitHub dispatch failed: ${resp.status} ${await resp.text()}`);
  }
}

export default {
  // ctx (ExecutionContext) lets us use ctx.waitUntil() to fire the GitHub
  // dispatch *after* returning 200 to Slack. Without this, Slack retries the
  // webhook if GitHub is slow, creating duplicate workflow runs.
  async fetch(request, env, ctx) {
    if (request.method !== "POST") return new Response("OK");

    // Read raw body text — required for signature verification (must use the
    // exact bytes Slack sent, not a re-serialised JSON object).
    const bodyText = await request.text();

    // Parse body early so we can handle the url_verification challenge.
    let body;
    try {
      body = JSON.parse(bodyText);
    } catch {
      return new Response("OK");
    }

    // --- URL verification challenge ---
    // Slack sends this once during Event Subscriptions setup. We must respond
    // with the challenge value. Slack DOES sign this request, so we verify
    // the signature first before echoing the challenge back.
    if (body.type === "url_verification") {
      const timestamp = request.headers.get("X-Slack-Request-Timestamp");
      const sig = request.headers.get("X-Slack-Signature");
      const valid = await verifySlackSignature(
        env.SLACK_SIGNING_SECRET,
        timestamp,
        bodyText,
        sig
      );
      if (!valid) return new Response("Unauthorized", { status: 401 });
      return Response.json({ challenge: body.challenge });
    }

    // --- Replay attack prevention ---
    // Reject requests with a timestamp more than 5 minutes old.
    const timestamp = request.headers.get("X-Slack-Request-Timestamp");
    const fiveMinutesAgo = Math.floor(Date.now() / 1000) - 300;
    if (!timestamp || parseInt(timestamp, 10) < fiveMinutesAgo) {
      return new Response("Request too old", { status: 401 });
    }

    // --- Signature verification ---
    const sig = request.headers.get("X-Slack-Signature");
    const valid = await verifySlackSignature(
      env.SLACK_SIGNING_SECRET,
      timestamp,
      bodyText,
      sig
    );
    if (!valid) return new Response("Unauthorized", { status: 401 });

    // --- Event filtering ---
    // Only process DM message replies (not top-level messages, not bot posts).
    const event = body.event;
    if (!event || event.type !== "message" || event.channel_type !== "im") {
      return new Response("OK");
    }
    // thread_ts present → this is a reply (not the opening message).
    // bot_id present → message was sent by a bot (skip our own replies).
    if (!event.thread_ts || event.bot_id) {
      return new Response("OK");
    }

    // --- KV check: is this thread_ts a known pending recommendation? ---
    // The nightly GHA workflow populates PENDING_RECS_KV with all active
    // thread_ts keys after each run so this worker can filter quickly.
    const knownTs = await env.PENDING_RECS_KV.get(event.thread_ts);
    if (!knownTs) {
      console.log(`thread_ts ${event.thread_ts} not in pending recs — ignoring`);
      return new Response("OK");
    }

    const replyText = event.text || "";
    if (!replyText.trim()) {
      console.log(`Skipping thread ${event.thread_ts} — empty message text`);
      return new Response("OK");
    }
    console.log(`Dispatching reply for thread ${event.thread_ts}: "${replyText}"`);

    // Acknowledge immediately — Slack will not retry if we respond quickly.
    // The GitHub dispatch runs in the background after the response is sent.
    ctx.waitUntil(dispatchToGitHub(env, event.thread_ts, replyText));
    return new Response("OK");
  },
};
