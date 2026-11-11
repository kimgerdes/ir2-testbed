# Demo queries (bge-small, one vector, vs GTE-ModernColBERT, many vectors; BM25 for reference)
Document embeddings are precomputed, and the models load once, then each query takes about a second.
Each query shows the top 5 of the three systems, how many relevant documents each puts in its top 2, and where it ranks them.

## Part 1, the core: LIMIT-small (46 people, 1000 "Who likes X?" queries)

Run `python pairs/04-limit/scripts/demo.py`, then type the queries.

- **Hero**: "Who likes Fruit Snacks?" ColBERT and BM25 rank both people 1st and 2nd; bge-small ranks them 30th and 35th. The query is trivial for word matching, so the failure is not about understanding: one 384d vector per person cannot keep apart the pairs that 1000 such queries ask for.
- **Failure**: "Who likes Oil Painting?" ColBERT ranks Flor Lemaire, who likes Vegetable Oil and Watercolor Painting, 2nd, above Gladstone Oonk, who likes Oil Painting (3rd). Each query token takes its best match anywhere in the document, so "oil" and "painting" are matched in two different items. One of the 5 LIMIT-small queries where ColBERT misses someone in the top 2 (`demo.py --suggest` lists them).
- **Wildcard**: proposed on the spot. Any "Who likes X?" is judged live by reading the 46 documents (this reading reproduces all 1000 LIMIT-small qrels; singular and plural both work). If nobody likes X, the demo says so and lists close items to pick from. Backup: "Who likes Waffles?" (two people, not one of the dataset's queries).

## Part 2, the testbed: SciFact (5,183 abstracts, the setting of our row)

Run `python pairs/04-limit/scripts/demo.py --dataset scifact`, then type the queries.

- **Hero**: "Ivermectin is used to treat onchocerciasis." The labelled study (*"Rapid-Impact Interventions": How a Policy of Integrated Control for Africa's Neglected Tropical Diseases Could Benefit the Poor*) names onchocerciasis once, in a list of diseases. ColBERT and BM25 rank it 1st; bge-small 24th, behind drug-therapy abstracts. The same limit as in Part 1: one vector for the whole abstract loses a single rare detail that token matching keeps.
- **Failure**: "Epidemiological disease burden from noncommunicable diseases is more prevalent in low economic settings." bge-small ranks the labelled study (*Global, regional, and national comparative risk assessment of 79 … risks*) 1st; ColBERT 23rd, behind abstracts that match "epidemiology", "prevalence" and "burden" word by word. Token matching rewards the vocabulary, not the claim. The same thing at scale: ArguAna, −0.175 nDCG@10, where a counter-argument shares the topic words of every argument on the same debate (`scripts/analysis.py` lists the largest losses).
- **Wildcard**: a claim from the room. SciFact has no label for a new claim, so the room judges the top 5. Backup: "Aspirin reduces the risk of colorectal cancer."

## Fallback recording

Here is the link to a video recording of the demo: [LIMIT demo](https://drive.google.com/drive/folders/1q_bnUrGWC0wCWrM8ve2B_lhKwidf7c7T?usp=sharing).