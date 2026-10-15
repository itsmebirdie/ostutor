# Getting LLM access for OS Tutor

The code needs only an API key for any **OpenAI-compatible chat-completions endpoint**.
The plan's default is Groq (free tier, runs Llama 3.3 70B). Provider limits and sign-up
steps change, so treat the numbers below as things to confirm in the console.

## Option A: Groq (recommended, matches the plan)

1. Go to https://console.groq.com and sign up (Google/GitHub login works). No developer
   "approval" is needed; a free account can create keys immediately.
2. Open **API Keys** -> **Create API Key**, name it `ostutor`, and copy the key. It is shown
   once. It looks like `gsk_...`.
3. In the console, open the **Limits** page (wording may vary) and note the daily token
   cap for `llama-3.3-70b-versatile`. The report quotes 100,000 tokens/day from third-party
   summaries; replace that with what the console says.
4. In the repo:
   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   # edit .env and paste the key after GROQ_API_KEY=
   python -m ostutor.cli ping  # should print "LLM access works."
   ```
5. Try it:
   ```bash
   python -m ostutor.cli analyze data/programs/deadlock_two_locks.ost --llm --db cache.sqlite
   ```
   Run the same command again: the second run uses the cache and costs 0 tokens.

Never paste the key into code, chat, screenshots or a commit. If it leaks, delete it in the
console and create a new one.

## Option B: other free or cheap providers (no code change)

Set `OSTUTOR_BASE_URL`, `OSTUTOR_MODEL` and `OSTUTOR_API_KEY` in `.env`:

| Provider | Where to get a key | Base URL |
|---|---|---|
| Google Gemini | https://aistudio.google.com (Get API key) | `https://generativelanguage.googleapis.com/v1beta/openai` |
| OpenRouter | https://openrouter.ai (Keys; some models are free) | `https://openrouter.ai/api/v1` |
| Cerebras | https://cloud.cerebras.ai | `https://api.cerebras.ai/v1` |

Model names differ per provider; copy the exact id from the provider's model list. If a
provider rejects JSON mode, the client automatically retries without it, and the checker
still validates the output. The plan lists a second provider as overflow for the daily cap.

## Option C: Anthropic API (paid)

Create an account at https://console.anthropic.com, add billing credit, and create a key under
API keys. Anthropic's endpoint is not OpenAI-format by default, so this repo's client does not
call it directly; use it only if you add a small adapter in `narration/llm.py`.
Check https://docs.claude.com for current model ids and pricing.

## Spending tokens wisely (the daily cap is the binding limit)

- Tests never call the network: they use a fake transport.
- While developing, run with `--fixtures tests/fixtures_llm`. The first run records each
  response; later runs replay it for free (`--fixture-mode replay` guarantees no network).
- Use `--db cache.sqlite` so repeated questions cost nothing.
- Each narration call sends roughly 300-600 tokens of evidence plus the reply, so the
  100-call/day estimate in the report is about right; measure `tokens:` printed by the CLI
  and report that instead of the estimate.
- If the key is missing, wrong, rate-limited or out of quota, the tutor shows the template
  narration with a "template fall-back" badge instead of failing (design rule 5).
