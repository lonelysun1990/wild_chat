# Transactional intent analysis (0_10000 labeled data)

## Summary

Review of the **128 conversations** labeled as `transactional` in `intent_output/0_10000/` shows that **your statement is correct**: a large share are mislabeled. Many are coding-related prompts, writing tasks under specific constraints, or content-generation requests—not commercial purchase or booking actions.

## Findings

### 1. Confirmed mislabels (should not be transactional)

- **Agent / system prompts (CONSTRAINTS + COMMANDS)**  
  Dozens of nearly identical first messages that define memory limits, command lists (`google`, `memory_add`, `write_to_file`, `start_agent`, etc.). These are **coding/task setup** (informational → coding or casual_other), not purchase or booking intent.

- **“Create / write / build / generate” requests**  
  Examples: “create the files mentioned in this manifest.json”, “build free access to gpt4 app for me”, “create an ai image of a cat”, “write a bid on this … Full Stack developer”, “write a message for an air conditioning company”, “write a formal letter … Remind him about the weekly payment”, “write a attractive feed post for lazada … voucher”.  
  These are **informational** (coding or creative_writing), not the user performing a transaction.

- **Tool / API usage**  
  “use aspose-words convert word to pdf”, “use aspose-cells convert excel to pdf” → **informational (coding)**.

- **Roleplay / scenario descriptions**  
  “A Man whose English is his third language tries to order something at a restaurant” describes a scenario; it is not “I want to order now” → **informational (education/casual_other)**.

- **Single-word or config snippets**  
  “approve”, or `cmd=delete path=...` style config → **informational (coding/support)** or unclear.

- **Consultation / job content**  
  “Provide Consultation Regarding Deep Learning Robotics Project”, internship description, cover letter for job posting → **informational** (education/creative_writing).

- **Rephrase / draft tasks**  
  “Please rephrase Hello Walmart Team …”, “你是一个客服收到一封客户来信…” (draft a customer service reply) → **informational (creative_writing)**.

### 2. Why the model over-applied “transactional”

- The original guideline said: *“transactional: User is buying or **taking an action** (buy, order, purchase, cart, checkout, pay, sign up, book now, reserve).”*
- **“Taking an action”** is too broad. The model treated “create”, “write”, “build”, “generate”, “convert”, “submit”, “approve” as user “actions,” and sometimes conflated “sign up” with agent “sign up”/command wording.
- No explicit **exclusions**, so task-completion and content-generation requests were folded into transactional.

### 3. What should stay transactional

- Queries that clearly express **completing a commercial or service transaction**: buy, order, pay, checkout, add to cart, book now, reserve, sign up for a paid event/service, refund, return, get a ticket, etc.
- Example that fits: “Get your kahk from cafe cornish and get the finest kahk in town” (promotional but purchase-oriented).
- “cheapest ticket from manila to bremen” is borderline (could be commercial_investigation or transactional depending on framing).

## Guideline change (applied in code)

Transactional is narrowed to **real-world commercial or service transactions**: the user intends to **complete a purchase, payment, booking, reservation, or registration for a paid service/event**.  

Explicitly **exclude** from transactional:

- Requests for the AI to **create, write, build, generate, convert, or draft** content (→ informational: coding or creative_writing).
- **System/agent prompts**, command lists, and task constraints (→ informational: coding or casual_other).
- **Roleplay or scenario descriptions** (e.g. “a man tries to order at a restaurant”) (→ informational: education/casual_other).
- **Single-word or technical snippets** (e.g. “approve”, config blocks) unless clearly purchase/booking (→ informational or support).

This is implemented in `commercial_vertical_utils.py` via updated `INTENT_GUIDELINES`.
