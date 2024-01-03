import copy
import pickle as pkl

import spacy
from datasets import concatenate_datasets, load_dataset
from NRCLex.nrclex import NRCLex

dataset_name = "xsum"
doc_summ_keys = {"document": "document", "summary": "summary"}
version = None
batch_size = 128
nrc = NRCLex("./NRCLex/nrc_en.json")
lemmatize_nrc = True

if lemmatize_nrc:
    nlp = spacy.load("en_core_web_sm")

    lexicon_words = list(nrc.__lexicon__.keys())
    lemmatized_lexicon = {}

    for lexicon_word in nlp.pipe(lexicon_words):
        lexicon_lemma = lexicon_word[0].lemma_

        if lexicon_lemma not in lemmatized_lexicon:
            lemmatized_lexicon[lexicon_lemma] = nrc.__lexicon__[
                lexicon_word.text
            ]

    nrc.__lexicon__ = lemmatized_lexicon

dataset = load_dataset(dataset_name, version=version)
all_dataset = concatenate_datasets(
    [dataset["train"], dataset["test"], dataset["validation"]]
)

documents = [doc for doc in all_dataset[doc_summ_keys["document"]]]
ref_summaries = [ref_summ for ref_summ in all_dataset[doc_summ_keys["summary"]]]

output_dict = {
    "nrclex_documents": [],
    "nrclex_summaries": [],
}

for i in range(0, len(documents), batch_size):
    print("Samples processed:", i)
    batch_documents = documents[i : i + batch_size]
    batch_ref_summaries = ref_summaries[i : i + batch_size]

    batch_nrclex_documents = []
    batch_nrclex_ref_summaries = []

    if lemmatize_nrc:
        for document in nlp.pipe(
            [document.lower() for document in batch_documents],
            n_process=4,
            disable=["parser", "ner"],
        ):
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(" ".join([token.lemma_ for token in document]))
            batch_nrclex_documents.append(nrclex)

        for ref_summary in nlp.pipe(
            [ref_summary.lower() for ref_summary in batch_ref_summaries],
            n_process=4,
            disable=["parser", "ner"],
        ):
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(
                " ".join([token.lemma_ for token in ref_summary])
            )
            batch_nrclex_ref_summaries.append(nrclex)

    else:
        for document in batch_documents:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(document.lower())
            batch_nrclex_documents.append(nrclex)

        for ref_summary in batch_ref_summaries:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(ref_summary.lower())
            batch_nrclex_ref_summaries.append(nrclex)

    output_dict["nrclex_documents"] += batch_nrclex_documents
    output_dict["nrclex_summaries"] += batch_nrclex_ref_summaries


fname = "%s-nrclex" % (dataset_name)

if lemmatize_nrc:
    fname += "+lemmatized"

with open("%s.pkl" % fname, "wb") as fw:
    pkl.dump(output_dict, fw)
