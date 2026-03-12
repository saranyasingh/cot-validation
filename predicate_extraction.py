import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Premise 1: The victim was found dead in the study, so the murder occurred in the study. (fact from the story)

Premise 2: The killer must have had access to the study and a weapon to commit the murder. (definition of murder location and method)

Premise 3: Clara saw Bob leave the study shortly before the murder, which establishes Bob’s proximity to the study around the time of the murder. (fact from the story)

Premise 4: Bob was in the garden holding a knife, so Bob possessed a weapon at that time. (fact from the story)

Premise 5: Therefore, Bob had both the opportunity (near the study, leaving it shortly before the murder) and the weapon (knife) to commit the murder. (combination of Premises 3 and 4)

Premise 6: There is no stated evidence that Alice or Clara had the same combination of proximity to the study and weapon at that time. (fact about the absence of contrary evidence)

Conclusion: The murderer is Bob.

"""

doc = nlp(text)

print("=== TOKENS AND POS TAGS ===")
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

print("\n=== SENTENCE ANALYSIS ===")

def extract_predicates(sent):
    predicates = []

    for token in sent:
        # find verbs (potential predicates)
        if token.pos_ == "VERB":
            subj = None
            obj = None
            loc = None
            time = None

            # look for subject/object dependencies
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child.text
                if child.dep_ in ("dobj", "attr", "pobj"):
                    obj = child.text
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            loc = grandchild.text

            predicates.append({
                "verb": token.lemma_,
                "subject": subj,
                "object": obj,
                "location": loc
            })

    return predicates


for sent in doc.sents:
    print("\nSentence:", sent.text)

    preds = extract_predicates(sent)

    for p in preds:
        print("Candidate predicate:", p)