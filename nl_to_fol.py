from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "fvossel/t5-base-nl-to-fol"  # small and fast

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = '''
Premise 1: Mack was killed at the bungee site by a nunchaku, so the murderer must have had access to a nunchaku and the ability to use it.

Premise 2: Mackenzie has explicit martial arts training with nunchaku and has a recent history of owning or handling a nunchaku (security notes show him buying one a week before, and there is a bruise on his knuckles from training), establishing both capability and access.

Premise 3: There are witnesses who report Mackenzie and Mack arguing at the bungee site, and Mackenzie voiced disparaging remarks about Mack, indicating a motive rooted in jealousy and rivalry for status or recognition.

Premise 4: The forensic evidence identifies a nunchaku matching the murder weapon, and the item is described as something kept but not valued highly by its owner, consistent with a weapon the owner used and possibly discarded in a moment of impulse—this links the weapon to the person who owns or uses such a weapon.

Premise 5: Ana’s involvement is less supported by direct link to the murder weapon or motive; she did not go through with the jump and lacks a direct access path to the weapon, whereas Mackenzie fits the weapon-access-motive triangle more tightly.

Conclusion: Given a murder weapon (nunchaku) tied to the capable owner, plus a clear motive from jealousy and corroborating on-site argument, Mackenzie is the most likely murderer.
'''

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=400)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))