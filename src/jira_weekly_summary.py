#!/usr/bin/env python3
"""Jira weekly account summary agent for CPM — Slack delivery."""

import os
import json
from datetime import datetime, timezone
import requests
import anthropic

# ── Config ───────────────────────────────────────────────────────────────────
JIRA_BASE_URL  = "https://skyflow.atlassian.net"
JIRA_EMAIL     = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
FILTER_ID      = "12006"

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_USER_ID   = os.environ["SLACK_USER_ID"]   # e.g. U01234567

# ── Jira helpers ──────────────────────────────────────────────────────────────

def jira_get(path, params=None):
    resp = requests.get(
        f"{JIRA_BASE_URL}/rest/api/3{path}",
        auth=(JIRA_EMAIL, JIRA_API_TOKEN),
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def extract_adf_text(node):
    """Recursively extract plain text from Atlassian Document Format (ADF)."""
    if not node or not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = [extract_adf_text(child) for child in node.get("content", [])]
    separator = " " if node.get("type") in ("doc", "paragraph", "heading") else ""
    return separator.join(p for p in parts if p).strip()


def get_filter_issues():
    """Fetch all issues from the Jira filter, paginating as needed."""
    issues, start_at, max_results = [], 0, 100
    fields = [
        "summary", "status", "assignee", "description", "comment",
        "priority", "labels", "project", "issuetype", "updated",
        "created", "subtasks", "issuelinks", "components",
    ]
    while True:
        data = jira_get("/search", params={
            "jql": f"filter={FILTER_ID} ORDER BY updated DESC",
            "startAt": start_at,
            "maxResults": max_results,
            "fields": ",".join(fields),
        })
        issues.extend(data["issues"])
        if start_at + max_results >= data["total"]:
            break
        start_at += max_results
    return issues


def format_issue(issue):
    """Convert a raw Jira issue into a clean dict for the prompt."""
    f = issue["fields"]

    raw_comments = (f.get("comment") or {}).get("comments", [])
    comments = []
    for c in raw_comments[-5:]:
        author = (c.get("author") or {}).get("displayName", "Unknown")
        body = extract_adf_text(c.get("body") or {})[:400]
        date = (c.get("updated") or "")[:10]
        if body:
            comments.append(f"[{date}] {author}: {body}")

    links = []
    for lnk in f.get("issuelinks") or []:
        for direction in ("outwardIssue", "inwardIssue"):
            li = lnk.get(direction)
            if li:
                links.append({
                    "key": li["key"],
                    "url": f"{JIRA_BASE_URL}/browse/{li['key']}",
                    "summary": li["fields"].get("summary", ""),
                    "status": li["fields"]["status"]["name"],
                    "relationship": lnk["type"].get(
                        "outward" if direction == "outwardIssue" else "inward", ""
                    ),
                })

    subtasks = []
    for st in (f.get("subtasks") or [])[:10]:
        subtasks.append({
            "key": st["key"],
            "url": f"{JIRA_BASE_URL}/browse/{st['key']}",
            "summary": st["fields"].get("summary", ""),
            "status": st["fields"]["status"]["name"],
        })

    return {
        "key":             issue["key"],
        "url":             f"{JIRA_BASE_URL}/browse/{issue['key']}",
        "summary":         f.get("summary", ""),
        "status":          (f.get("status") or {}).get("name", "Unknown"),
        "priority":        (f.get("priority") or {}).get("name", ""),
        "project":         (f.get("project") or {}).get("name", ""),
        "assignee":        (f.get("assignee") or {}).get("displayName", "Unassigned"),
        "description":     extract_adf_text(f.get("description") or {})[:800],
        "recent_comments": comments,
        "linked_issues":   links[:10],
        "subtasks":        subtasks,
        "updated":         (f.get("updated") or "")[:10],
        "created":         (f.get("created") or "")[:10],
        "labels":          f.get("labels") or [],
        "components":      [c.get("name", "") for c in (f.get("components") or [])],
    }


# ── Claude summary ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are generating a weekly account summary for Travis, a Customer Program Manager (CPM) at \
Skyflow — a data privacy and tokenization company. Travis manages customer relationships and \
needs a clear, scannable digest to understand portfolio health, prioritize his week, and identify \
where help is needed.

Guidelines:
- Write for a CPM audience: business impact first, light on deep technical detail
- Assign a health indicator: 🟢 On Track · 🟡 Needs Attention · 🔴 At Risk
- Infer health from ticket statuses, priorities, comment sentiment, overdue items, and activity
- If an account has no recent activity, note it as quiet/stable
- Keep each section tight — this is a scan document, not a deep-dive

Output ONLY a valid JSON object — no markdown fences, no prose outside the JSON.
Schema:
{
  "executive_summary": "<2-3 sentence overview of overall portfolio health>",
  "accounts": [
    {
      "name": "<account name>",
      "health": "<🟢 | 🟡 | 🔴>",
      "current_status": "<1-2 sentences>",
      "next_steps": ["<step>", ...],
      "help_needed": ["<item>"],   // empty array if none
      "recent_tickets": [
        {"key": "<KEY-123>", "url": "<url>", "status": "<status>"}
      ]
    }
  ]
}
Keep next_steps to 3-4 items. Keep recent_tickets to the most relevant 5 or fewer."""


def generate_summary(issues_data: list) -> dict:
    """Call Claude and return the parsed summary dict."""
    client = anthropic.Anthropic()
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    user_content = (
        f"Today is {today}. Generate the weekly account summary for the following "
        f"Jira data:\n\n```json\n{json.dumps(issues_data, indent=2, default=str)}\n```"
    )

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        message = stream.get_final_message()

    for block in message.content:
        if block.type == "text":
            return json.loads(block.text)

    raise RuntimeError("Claude returned no text block in the response")


# ── Slack delivery ────────────────────────────────────────────────────────────

def _slack_post(endpoint: str, payload: dict) -> dict:
    resp = requests.post(
        f"https://slack.com/api/{endpoint}",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error on {endpoint}: {data.get('error')}")
    return data


def build_blocks(summary: dict, today: str) -> list:
    """Convert the summary dict into Slack Block Kit blocks."""
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📋 Weekly Account Summary — {today}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": summary["executive_summary"]},
        },
        {"type": "divider"},
    ]

    for acct in summary.get("accounts", []):
        # Account header line
        name_line = f"{acct['health']} *{acct['name']}*"

        # Body text — built in sections to stay under Slack's 3000-char block limit
        status_text = f"*Current Status:* {acct['current_status']}"

        steps = acct.get("next_steps") or []
        steps_text = "*Next Steps:*\n" + "\n".join(f"• {s}" for s in steps) if steps else ""

        risks = acct.get("help_needed") or []
        risks_text = (
            "*Help Needed / Risks:*\n" + "\n".join(f"• {r}" for r in risks)
            if risks
            else "*Help Needed / Risks:* None"
        )

        tickets = acct.get("recent_tickets") or []
        if tickets:
            ticket_lines = "\n".join(
                f"• <{t['url']}|{t['key']}> · {t['status']}" for t in tickets
            )
            tickets_text = f"*Recent Tickets:*\n{ticket_lines}"
        else:
            tickets_text = ""

        body_parts = [p for p in [status_text, steps_text, risks_text, tickets_text] if p]
        body_text = "\n\n".join(body_parts)

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{name_line}\n\n{body_text}"[:3000]},
        })
        blocks.append({"type": "divider"})

    return blocks


def send_slack_dm(summary: dict):
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # Open (or reuse) a DM channel with the user
    dm = _slack_post("conversations.open", {"users": SLACK_USER_ID})
    channel_id = dm["channel"]["id"]

    blocks = build_blocks(summary, today)

    # Slack allows max 50 blocks per message; split if needed
    chunk_size = 50
    for i in range(0, len(blocks), chunk_size):
        _slack_post("chat.postMessage", {
            "channel": channel_id,
            "text": f"Weekly Account Summary — {today}",  # fallback for notifications
            "blocks": blocks[i : i + chunk_size],
        })

    print(f"Summary sent to Slack DM for user {SLACK_USER_ID}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    print(f"Fetching issues from Jira filter {FILTER_ID}…")
    raw_issues = get_filter_issues()
    print(f"Found {len(raw_issues)} issues")

    issues_data = [format_issue(i) for i in raw_issues]

    print("Generating summary with Claude…")
    summary = generate_summary(issues_data)

    print("Sending to Slack…")
    send_slack_dm(summary)
    print("Done.")


if __name__ == "__main__":
    main()
