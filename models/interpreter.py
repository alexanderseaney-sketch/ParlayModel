"""
LLM write-ups for picks and entries (added 2026-10-02): turns the numbers the
+EV Finder already computed -- model vs de-vigged market probability per leg, the
model's real-line recompute, recent form, the prop model's top feature drivers,
same-team correlation adjustments, joint probability, payout and EV -- into a short
plain-English read: why each leg is priced the way it is, which leg is most likely
to sink the entry, and whether the entry is worth playing.

Claude only explains numbers computed here; it is told not to invent stats and
every figure it sees comes from this project's own data. No key -> the dashboard
shows the deterministic summary (template_summary) instead.

Key: ANTHROPIC_API_KEY (env / .env) or Streamlit secrets.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are the analyst inside an NFL player-prop betting dashboard.
You receive a JSON description of a pick'em entry: each leg's line, the side taken,
the project's model probability, the de-vigged market probability, the edge, recent
game logs, the prop model's most important features, and the entry-level joint
probability, payout and expected value.

Write a short read for a bettor who knows the basics:
1. One line per leg: the main reason the model is on that side (cite the numbers
   given -- recent form vs the line, the model's drivers), and anything that makes it
   fragile (thin edge, volatile stat, small sample, injury status).
2. **Riskiest leg:** name the single leg most likely to sink the entry and why.
3. **Verdict:** play / pass / trim, in one or two sentences, grounded in the EV and
   joint probability.

Rules: use only numbers present in the JSON -- never invent stats, injuries or news.
If a leg has no market probability, say its edge is unverified. Keep it under 220
words. Markdown, no tables."""


def api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def build_context(entry: dict, legs: list[dict]) -> dict:
    """Plain-JSON context. `legs` items come from the +EV Finder's candidate table."""
    def r(x, n=3):
        try:
            return round(float(x), n)
        except (TypeError, ValueError):
            return None

    return {
        "entry": {
            "n_legs": entry.get("n_legs"),
            "joint_probability_model": r(entry.get("joint_prob")),
            "joint_probability_if_independent": r(entry.get("naive_prob")),
            "joint_probability_market": r(entry.get("market_joint_prob")),
            "payout_multiple": r(entry.get("payout"), 2),
            "expected_value_per_dollar": r(entry.get("ev")),
            "correlated_pairs": [{"a": a, "b": b, "phi": r(phi)} for a, b, phi in entry.get("adjustments", [])],
        },
        "legs": [{
            "player": l.get("player"), "team": l.get("team"), "opponent": l.get("opponent"),
            "stat": l.get("stat_name"), "line": l.get("line"), "side": str(l.get("choice")).upper(),
            "model_probability": r(l.get("prob")), "market_probability": r(l.get("market_prob")),
            "edge": r(l.get("edge")), "decimal_price": r(l.get("decimal"), 2),
            "model_baseline_proxy_line": r(l.get("proxy_line"), 1),
            "recent_games": l.get("recent_games"),
            "injury_status": l.get("injury_status"),
            "top_model_features": l.get("top_features"),
        } for l in legs],
    }


def template_summary(entry: dict, legs: list[dict]) -> str:
    """Deterministic fallback when no API key is configured."""
    from parlay_calculator import riskiest_leg
    lines = []
    for l in legs:
        mk = l.get("market_prob")
        mk_txt = f"market {mk*100:.0f}%" if mk is not None and mk == mk else "no market price"
        lines.append(f"- **{l['player']}** {str(l['choice']).upper()} {l['line']} {l['stat_name']}: "
                     f"ours {l['prob']*100:.0f}% vs {mk_txt}")
    risk = riskiest_leg(legs)
    if risk:
        lines.append(f"\n**Riskiest leg:** {risk['player']} {str(risk['choice']).upper()} {risk['line']} "
                     f"{risk['stat_name']} (lowest hit probability, {risk['prob']*100:.0f}%).")
    if entry.get("ev") is not None:
        verdict = "positive" if entry["ev"] > 0 else "negative"
        lines.append(f"**EV:** {entry['ev']*100:+.1f}% per $1 ({verdict}); joint probability "
                     f"{entry['joint_prob']*100:.1f}% at {entry['payout']:.2f}x.")
    return "\n".join(lines)


def explain_entry(entry: dict, legs: list[dict]) -> tuple[str, str]:
    """Returns (markdown, source) where source is "claude" or "template"."""
    key = api_key()
    if not key:
        return template_summary(entry, legs), "template"
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    context = build_context(entry, legs)
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            # Server-side refusal fallback: if this model declines, the API re-runs
            # the request on a fallback model chosen by refusal category.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(context, indent=1)}],
        )
    except anthropic.AuthenticationError:
        return template_summary(entry, legs) + "\n\n_(Claude unavailable: invalid ANTHROPIC_API_KEY.)_", "template"
    except anthropic.RateLimitError:
        return template_summary(entry, legs) + "\n\n_(Claude unavailable: rate limited, try again shortly.)_", "template"
    except anthropic.APIStatusError as e:
        return template_summary(entry, legs) + f"\n\n_(Claude unavailable: API error {e.status_code}.)_", "template"
    except anthropic.APIConnectionError:
        return template_summary(entry, legs) + "\n\n_(Claude unavailable: network error.)_", "template"

    if response.stop_reason == "refusal":
        return template_summary(entry, legs), "template"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text or template_summary(entry, legs)), ("claude" if text else "template")
