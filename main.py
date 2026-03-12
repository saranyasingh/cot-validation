
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STORY = '''\
In an adrenaline inducing bungee jumping site, Mack's thrill-seeking adventure came to a gruesome end by a nunchaku; now, it's up to Detective Winston to unravel the deadly secrets between Mackenzie and Ana.

Winston took a gulp of his black coffee, staring at the notes sprawled across his desk. A murder case at a bungee jumping site was definitely out of the ordinary. Today's victim was a young man named Mack, loud mouthed and cocky by all accounts.

Mack was bungee jumping the day he was killed. Oddly enough, according to the records, no one else was documented at the bungee jumping site that day, making this case even more peculiar. The first stop for the day was to visit one of Mack's housemates, a woman named Ana. They were seen leaving in the same vehicle from their shared housing complex the morning of the murder, and it was time for Winston to dig deeper.

As he pulled into the shared housing driveway, a nondescript car came into sight. He learned from neighbours that it was frequently used by multiple residents, but Ana had a peculiar interest in it. She would insist on driving whenever with a group of friends, later meticulously cleaning the car after each use. An idiosyncrasy of hers maybe, but a part of the puzzle nonetheless.

Winston knocked on the door, Ana opened it warily, twiddling a cleaning cloth and spray in her hands and greeted him with a nervous nod. Ana gets nervous and fidgets with the cleaner and cloth when questioned. Winston could sense palpable unease as he started asking her questions.

"Ana, did you not join Mack and the others for bungee jumping today?" Winston questioned, to which she responded, "I signed up to jump. But I didn't end up going through with it."

"Any particular reason you didn't join the others, Ana?" Winston proceeded.

Ana took a deep breath, "Well sir, my faith doesn't really permit bungee jumping. Truth be told, I was persuaded strongly by Mack. I had even signed up out of peer pressure but couldn't push myself."

It was true – Mack was insisting that everyone in the group should bungee jump. Mack had reportedly also been vocal about ridiculing Ana’s faith, even encouraging others to join him in doing so. It was a significant factor in their relationship.

"Ana, did you and Mack leave in the same car for the bungee jumping event this morning?" Winston gently pushed further.

"Yes. Yes, we did. We always carpool." She responded while anxiously using the cleaner and cloth on her car’s dashboard. Her eyes flickered nervously back to Winston, expecting the next question.

Winston took a deep breath, standing up to leave, "Alright Ana, that should cover everything for now. We'll be in touch."

Ana nervously nodded without looking up from her cleaning, wringing the cloth repeatedly as Winston walked away, left again with another piece to the enigmatic puzzle of Mack's murder.

The day was getting older and Winston was getting more tired, but the case was fresh, and he wasn't one to back down. He tugged on his coat as he approached the bashful teen waiting for him by the police station.

"Mackenzie, it is?" he asked, extending his hand.

"Yeah, that's right." The slight lisp, overlaid with blanket anxiety, confirmed what the school reports suggested.

"You were at the site when Mack... erm... you know," Winston's voice was methodical, calm -- almost robotic. The suspicion on Mackenzie was not unfounded - the security cameras showed him buying nunchaku a week before.

Mackenzie shifted on his feet, looking away before answering, "Yeah, I was there."

Winston pulled out a small notebook, "What were you doing there, Mackenzie?”

“Bungee jumping, like Mack… Then I left. I didn't... I didn't do anything…” Mackenzie replied.

Internally, Winston sighed at the never-ending waterfall of teenage angst this case was turning into.

“Martial arts, huh?” Winston segued, gesturing to a bruise on Mackenzie’s knuckles. “Nunchaku particularly, I see? Training does include the use of those, correct?”

The change in Mackenzie’s demeanor mirrored the bitterness in the last month’s weather – dark eyes replaced with ice-cold ones. “Yeah,” he admitted, shrinking slightly.

Mackenzie always took pride in being the best at everything. So when Mack got everything he wanted - the promotion to team captain, the respect, the attention - it was a hard pill for Mackenzie to swallow. Winston remembered the team talk, Mackenzie was indeed the top candidate but it had gone to Mack instead.

What clinched it was Mackenzie’s remarks about Mack, echoing whispers of dispute and bickering, lost in the crowded lunchroom. There were also multiple witness reports of the two seen arguing at the bungee jumping site previously. Mackenzie had indeed said disparaging, almost emotional things about Mack – all stemming from a potent brew of jealousy, Winston inferred.

Shifting later through the detritus of Mackenzie's life, Winston discovered the nunchaku that matched the forensics report. They were tucked away, but the layer of dust suggested they weren't a favored possession anymore. It wasn’t hidden, it was misplaced – discarded in the throes of developing maturity.

As the sun started to set, Winston could see witnesses, scattered across the park, repeatedly pointing to the bungee jumping scaffolding. It occurred to him, then, the narrative of the past days. Mackenzie, jealous and wronged, over and over, at the same sight. It was quite a sight.

Winston, shuffling back to the station, was left with one thought - Looks like Mackenzie had quite an eventful week.

'''

# ─── Stage 1: Chain-of-Thought Reasoning ─────────────────────────────────────

cot_reasoning = client.responses.create(
    model="gpt-5-nano",
    input=f'''
I will give you a murder story. Provide a logical argument for who is the most likely murderer. Explain every step of your logical reasoning in the format of a logical syllogism where each premise follows from the previous one, with enough context where I can understand your logic even without the full story. End your answer with The murderer is: X where X is your final answer.

{STORY}
'''
)

cot_text = cot_reasoning.output_text
print("=== Chain of Thought Reasoning ===")
print(cot_text)

story_path = os.path.join(SCRIPT_DIR, "story.txt")
with open(story_path, "w") as f:
    f.write(STORY)

# ─── Stage 2: Structured FOL Extraction ──────────────────────────────────────

STRUCTURED_FOL_TEMPLATE = '''\
Convert the following chain-of-thought reasoning into structured first-order logic (FOL).

=== SYNTAX ===
- Quantifiers: ∀ (universal), ∃ (existential)
- Connectives: ∧ (and), ∨ (or), → (implies), ¬ (not)
- Predicates: CamelCase (e.g., InRoom, Holding, FoundDead, MurderLocation)
- Constants: lowercase with underscores (e.g., alice, bob, mr_black, kitchen)
- Variables: single lowercase letters (x, y, z, w, t) — only in quantified formulas

=== TIME REPRESENTATION ===
- Use 24-hour time constants: t_21 for 9 PM, t_14 for 2 PM, etc.
- Use named time constants (t_murder, t_seen) ONLY when the exact time is genuinely unspecified.
- Temporal predicates: Before(t1, t2), ShortlyBefore(t1, t2)

=== OUTPUT CATEGORIES ===
Produce exactly three sections:

## FACTS
Ground atomic formulas taken directly from the story. No quantifiers.
Each fact is one atomic predicate applied to constants.

## RULES
Universally quantified implications that encode external or linguistic knowledge.
Every rule MUST start with ∀. These are general principles, not story-specific.

## INFERENCES
Derived conclusions. Each inference MUST cite the FACT/RULE/INF labels it depends on
and name the inference rule used (Modus Ponens, Universal Instantiation, etc.).

=== OUTPUT FORMAT ===
Each line follows this format:
LABEL: <natural language gloss> :: <FOL formula>

Use :: as the delimiter between the gloss and the formula.
Labels are FACT-1, FACT-2, ..., RULE-1, RULE-2, ..., INF-1, INF-2, ...

=== CONSTRAINTS ===
- Do NOT merge a fact and a rule into one formula.
- Do NOT use vague time tokens like "deathtime" — use explicit constants or named constants.
- Every inference MUST cite its premises by label.
- Output ONLY the three sections with labeled lines. No extra explanation.

=== EXAMPLE ===
Given a story: "It was raining at noon. If it rains, the ground is wet."

## FACTS
FACT-1: It was raining at noon :: Raining(t_12)

## RULES
RULE-1: If it rains at a time, the ground is wet at that time :: ∀t (Raining(t) → GroundWet(t))

## INFERENCES
INF-1: From FACT-1, RULE-1 by Modus Ponens: The ground was wet at noon :: GroundWet(t_12)

=== CHAIN OF THOUGHT REASONING TO CONVERT ===
{cot_text}
'''

structured_prompt = STRUCTURED_FOL_TEMPLATE.format(cot_text=cot_text)
structured_response = client.responses.create(model="gpt-5-nano", input=structured_prompt)

structured_fol = structured_response.output_text
print("\n=== Structured FOL ===")
print(structured_fol)

# Save intermediate output so tptp.py can pick it up
fol_path = os.path.join(SCRIPT_DIR, "structured_fol.txt")
with open(fol_path, "w") as f:
    f.write(structured_fol)
print(f"Wrote {fol_path}")
