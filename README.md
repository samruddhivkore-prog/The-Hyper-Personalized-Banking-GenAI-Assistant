# Aisling Bank Assistant

A chat assistant for a (fictional) Irish bank. A customer can ask it a
banking question, check what they've spent, get a suggestion for the right
product, or freeze a lost card — and everything it tells them is pulled from
the bank's real policy pages, not invented on the spot.

> For the engineering write-up (architecture, test coverage, the bugs I
> found and fixed along the way), see [TECHNICAL.md](TECHNICAL.md). This
> page is the plain-English version.

## What it actually does

Open the chat and ask it things like:

- *"What's the fastest way to send money?"* → a real answer, with the fee
  and how long it takes, pulled from the bank's actual transfer policy.
- *"How much did I spend in August?"* → a one-line summary and a pie chart
  of where the money went, worked out from real transaction records — the
  assistant never does this maths itself, it only reads the answer back.
- *"Which account is best for saving?"* → a recommendation, in plain
  language, picked from the bank's real product list.
- *"Freeze my card"* → it asks "freeze your card ending 4821?" and does
  **nothing** until you say yes. No button, no change.
- *"Do you offer car insurance?"* → an honest "I couldn't find this in our
  policies," because the bank genuinely doesn't offer that — it never makes
  up an answer to avoid saying "I don't know."

There's also a separate, locked-down screen for bank staff that shows the
behind-the-scenes detail a customer never sees: why each customer was shown
a particular product, and a fairness check confirming the assistant isn't
quietly favouring one group of customers over another.

## Why I built this

Banks are leaning hard into AI for customer service and product
recommendations right now, and that comes with two problems that actually
matter, not just technical curiosities: the assistant can confidently make
things up about a real financial product, and a recommendation engine can
end up treating customer groups unfairly without anyone noticing.

I wanted to build something that treated both of those as engineering
requirements from day one, not as afterthoughts bolted on once it was
"working" — partly because that's exactly the gap I've seen between a
pretty AI demo and something a regulated business could actually trust, and
partly because it let me carry fairness-auditing ideas from my own prior
research work into an actual working system in a different domain. The
fairness check in this project isn't decorative: while building it, it
genuinely caught a real, unintended bias in my own recommendation logic
(see [TECHNICAL.md](TECHNICAL.md) for the specific finding) — which is the
whole point of having one.

*(This paragraph is my best guess at your motivation based on the project's
own design notes — swap it for your own words before this goes in front of
anyone.)*

## What it looks like

I don't have a screen-recording tool available to capture a real screenshot
right now, but here's an actual conversation with the running assistant,
copy-pasted from a live test — nothing in this transcript is made up:

```
You: What's the fastest way to send money to my sister?

Assistant: There is no fee to send or receive a standard SEPA transfer.
Standard SEPA Credit Transfers typically settle by the end of the next
business day. Personal accounts may send up to €10,000 per day and €25,000
per rolling 7-day period.
Source: SEPA Credit Transfer

────────────────────────────────────────────────────

You: How much did I spend in August?

Assistant: You spent €1,648 between 1 Aug and 31 Aug. Most went on bills
(€609). That is 78% more than the period before.
[a pie chart of spending by category appears here in the app]

────────────────────────────────────────────────────

You: Freeze my card

Assistant: Freezing is free, instant, and reversible if the card turns up
again.
                                    Freeze your card ending 3315?
                                    [ Yes ]   [ No ]

You: *clicks Yes*

Assistant: Your card ending 3315 has been frozen.

────────────────────────────────────────────────────

You: Do you offer car insurance?

Assistant: I couldn't find this in our policies. Please contact the bank
directly.
```

The actual screen also shows a "Recommended for you" strip of three product
cards above the chat, and a customer ID badge in the corner so it's obvious
which (demo) customer you're looking at.

## Known issues and limitations

Being upfront about what isn't polished yet:

- **A couple of tricky questions only work correctly with a real AI model
  switched on**, not the free offline mode. Telling "a personal loan" apart
  from "a mortgage," for example, needs actual judgement that the free
  fallback mode can't always get right — it occasionally mixes up two
  similar-sounding products. This is explained in detail, including the
  exact cases, in [TECHNICAL.md](TECHNICAL.md).
- **It isn't deployed anywhere public yet.** It runs fully on your own
  computer; putting it on the internet with a shareable link is the main
  thing left to do (see To-do below).
- **All the customers, transactions, and even the bank itself are made
  up.** This was a deliberate choice so the project needed no real customer
  data, but it does mean the numbers aren't from a real bank.
- **The free "Admin" screen for staff has no password by default** — it
  just shows a warning banner instead. A password can be turned on, but
  it's off unless you set one (see setup steps below).

## To-do

- [ ] Put a live, public version online so it can be tried without
      installing anything.
- [ ] Add a couple more product types to the catalogue.
- [ ] Make the tricky "which product is this?" questions reliable without
      needing a paid AI subscription.
- [ ] A friendlier, mobile-sized layout for the chat screen.

## How to run it yourself

You don't need to touch any file paths, and you don't need a paid
subscription to anything — everything below works out of the box on a
plain computer. These steps assume you've already installed
[Python](https://www.python.org/downloads/) (3.11 or newer) and have a
terminal open in the project folder.

**1. Create an isolated Python environment and install everything it
needs.** This keeps the project's dependencies separate from anything else
on your computer.

```bash
python -m venv .venv
```

Turn it on:

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

Install the project's requirements:

```bash
pip install -r requirements.txt
```

> If this step fails specifically on a package called `chromadb`, that's
> fine — skip it. That package needs a compiler toolchain some computers
> (especially Windows ones) don't have installed, and the project
> automatically uses a simpler built-in substitute if it's missing. Nothing
> else is affected.

**2. Create some demo data.** This invents 300 fake customers (with fake
transactions, a fake card each, and so on) and loads them into a small
local database file that gets created automatically in the project folder
— no separate database software needed.

```bash
python -m scripts.seed_db --n 300
```

**3. Load the policy documents** the assistant answers from:

```bash
python -m scripts.index_kb
```

(This step actually happens automatically the first time the app starts
too, so it's safe to skip — running it by hand just confirms it works.)

**4. Start the backend** (the part that does the actual thinking):

```bash
uvicorn app.main:app --reload
```

Leave this running. You should see a line ending in
`Uvicorn running on http://127.0.0.1:8000`.

**5. In a new terminal window**, with the same environment turned on again
(`.venv\Scripts\activate` or `source .venv/bin/activate`), start the
customer-facing screen:

```bash
streamlit run frontend/streamlit_app.py
```

A browser tab should open by itself at `http://localhost:8501`. If it
doesn't, open that address manually.

**6. (Optional) Start the staff-only admin screen**, in another new
terminal, same environment:

```bash
streamlit run frontend/admin_app.py --server.port 8502
```

Open `http://localhost:8502`. By default it opens with no password and a
warning banner. To add a password instead, set it before starting the app:

```bash
# Windows
$env:ADMIN_PASSWORD = "pick-something"
streamlit run frontend/admin_app.py --server.port 8502

# Mac/Linux
ADMIN_PASSWORD="pick-something" streamlit run frontend/admin_app.py --server.port 8502
```

### Using a real AI model instead of the free built-in one

By default, the assistant answers questions using a free, offline,
built-in stand-in — no account or payment needed, and this is what the
automated tests use. If you'd like it to use an actual AI model (which
answers more naturally and handles trickier questions better), get a free
API key from [Groq](https://console.groq.com/keys), then before starting
the backend:

```bash
# Windows
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key-here"

# Mac/Linux
export LLM_PROVIDER=groq
export GROQ_API_KEY=your-key-here
```

Then start the backend as in step 4. Google Gemini and Anthropic's Claude
are also supported the same way (`LLM_PROVIDER=gemini` / `anthropic` with
the matching key).

## How to run the automated tests

With the same environment turned on:

```bash
pip install -r requirements-dev.txt
pytest -v
```

This runs the full test suite using the free built-in stand-in model, so no
API key is required and nothing leaves your computer. If you've also set up
a real AI model as above, a handful of extra tests that specifically need
real judgement will run too, instead of being skipped.
