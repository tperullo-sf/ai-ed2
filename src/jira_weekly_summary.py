#!/usr/bin/env python3
"""Jira weekly account summary agent for CPM."""

import os
import json
import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
import anthropic

# ── Config ───────────────────────────────────────────────────────────────────
JIRA_BASE_URL      = "https://skyflow.atlassian.net"
JIRA_EMAIL         = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN     = os.environ["JIRA_API_TOKEN"]
FILTER_ID          = "12006"

GMAIL_USER         = os.environ["GMAIL_USER"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
RECIPIENT_EMAIL    = os.environ.get("RECIPIENT_EMAIL", "travis.perullo@skyflow.com")

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

    # Recent comments — last 5, most recent first
    raw_comments = (f.get("comment") or {}).get("comments", [])
    comments = []
    for c in raw_comments[-5:]:
        author = (c.get("author") or {}).get("displayName", "Unknown")
        body = extract_adf_text(c.get("body") or {})[:400]
        date = (c.get("updated") or "")[:10]
        if body:
            comments.append(f"[{date}] {author}: {body}")

    # Linked issues
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

    # Sub-tasks
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

Format guidelines:
- Write for a CPM audience: business impact first, light on deep technical detail
- Assign a health indicator to each account: 🟢 On Track · 🟡 Needs Attention · 🔴 At Risk
- Infer health from ticket statuses, priorities, comment sentiment, overdue items, and recent activity
- If an account has no recent activity, note it as quiet/stable rather than guessing at risks
- Keep each section tight — this is a scan document, not a deep-dive report
- Link tickets in Markdown: [KEY](url) · Status — no need to explain each ticket

Output must be valid HTML suitable for an email client (inline styles only, no external CSS).

Structure:
1. Executive summary paragraph at the top — 2–3 sentences covering overall portfolio health
2. One <section> per account:
   - <h2> with the account name and health indicator emoji
   - <p><strong>Current Status:</strong> 1–2 sentences</p>
   - <p><strong>Next Steps:</strong></p><ul>…3–4 bullet points…</ul>
   - <p><strong>Help Needed / Risks:</strong></p><ul>…bullets, or <em>None</em>…</ul>
   - <p><strong>Recent Tickets:</strong></p><ul>…linked list, key + status only…</ul>
3. <hr> between accounts
"""


def generate_summary(issues_data: list) -> str:
    """Call Claude to produce the HTML weekly summary."""
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
                "cache_control": {"type": "ephemeral"},  # cache stable system prompt
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        message = stream.get_final_message()

    for block in message.content:
        if block.type == "text":
            return block.text

    raise RuntimeError("Claude returned no text block in the response")


# ── Email ─────────────────────────────────────────────────────────────────────

_HTML_WRAPPER = """\
<!DOCTYPE html>
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;\
font-size:15px;line-height:1.6;color:#1a1a1a;max-width:820px;margin:0 auto;padding:24px;">
  <h1 style="color:#0052cc;border-bottom:2px solid #0052cc;padding-bottom:8px;margin-bottom:24px;">
    Skyflow Weekly Account Summary &mdash; {date}
  </h1>
  {body}
  <hr style="margin-top:40px;border:none;border-top:1px solid #ddd;">
  <p style="font-size:12px;color:#888;margin-top:8px;">
    Auto-generated from Jira filter 12006 &middot; {date}
  </p>
</body>
</html>
"""


def send_email(html_body: str):
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    subject = f"Weekly Account Summary — {today}"
    html_full = _HTML_WRAPPER.format(date=today, body=html_body)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText("Please view this email in an HTML-capable client.", "plain"))
    msg.attach(MIMEText(html_full, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())

    print(f"Summary sent to {RECIPIENT_EMAIL}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    print(f"Fetching issues from Jira filter {FILTER_ID}…")
    raw_issues = get_filter_issues()
    print(f"Found {len(raw_issues)} issues")

    issues_data = [format_issue(i) for i in raw_issues]

    print("Generating summary with Claude…")
    html_summary = generate_summary(issues_data)

    print("Sending email…")
    send_email(html_summary)
    print("Done.")


if __name__ == "__main__":
    main()
