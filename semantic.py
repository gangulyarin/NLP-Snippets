from nltk.corpus import wordnet as wn
import pandas as pd

term = 'fruit'
sn =wn.synsets(term)

print("Total synsets: ",len(sn))

sypd = pd.DataFrame([{'Definition': synset.definition(),
                      'Lexname': synset.lexname(),
                      'Lemmas': synset.lemma_names(),
                      'Examples': synset.examples()
                      } for synset in sn])

#print(sypd)

action_syn = wn.synsets('walk', pos="v")[0]
print(action_syn.entailments())

print(action_syn.lemmas()[0].antonyms()[0].synset().name())
print(action_syn.hyponyms()[0].name(), " - ", action_syn.hyponyms()[0].definition())

member_holonyms = wn.synsets('people')[0].member_holonyms()
print('Total Member Holonyms:', len(member_holonyms))
print('Member Holonyms :-')
for holonym in member_holonyms:
    print(holonym.name(), '-', holonym.definition())
    print()

part_meronyms = wn.synsets('wood')[0].part_meronyms()
print('Total Member Meronyms:', len(part_meronyms))
print('Member Meronyms :-')
for meronym in part_meronyms:
    print(meronym.name(), '-', meronym.definition())
    print()


hypernyms = wn.synsets('tree')[0].hypernyms()
print("Tree Hypernym : ",hypernyms)

# Check similarity based on hypernyms:
print("Similarity between cat and dog: ", wn.synsets('cat')[0].lowest_common_hypernyms(wn.synsets('dog')[0])[0].name())
