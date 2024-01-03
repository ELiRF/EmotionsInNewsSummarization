import copy
import json
import os

import spacy
from datasets import load_dataset
from NRCLex.nrclex import NRCLex


def get_emotion_chain(sample_nrclex):
    emotion_chain = []
    for sentence in sample_nrclex.sentences:
        emotions_list = []
        for word in sample_nrclex.affect_dict:
            if word in sentence:
                emotions_list.append(word)

        emotion_chain.append(" | ".join(emotions_list))

    return " ||| ".join(emotion_chain)


dataset_name = "xsum"
doc_summ_keys = {
    "document": "document",
    "summary": "summary",
}
version = None
batch_size = 128
modeling_datasets_path = "./ModelingDatasets"
model_name = "JES"

if not os.path.exists("%s/%s" % (modeling_datasets_path, model_name)):
    os.makedirs("%s/%s" % (modeling_datasets_path, model_name))

if not os.path.exists(
    "%s/%s/%s" % (modeling_datasets_path, model_name, dataset_name)
):
    os.makedirs("%s/%s/%s" % (modeling_datasets_path, model_name, dataset_name))

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


for partition in ["train", "validation"]:
    documents = [doc for doc in dataset[partition][doc_summ_keys["document"]]]
    ref_summaries = [
        ref_summ for ref_summ in dataset[partition][doc_summ_keys["summary"]]
    ]

    with open(
        "%s/%s/%s/%s.json"
        % (modeling_datasets_path, model_name, dataset_name, partition),
        "w",
    ) as fw:
        for i in range(0, len(documents), batch_size):
            print("Samples processed:", i)
            batch_documents = documents[i : i + batch_size]
            batch_ref_summaries = ref_summaries[i : i + batch_size]
            batch_nrclex_ref_summaries = []
            batch_emotion_chains = []

            if lemmatize_nrc:
                for ref_summary in nlp.pipe(
                    [
                        ref_summary.lower()
                        for ref_summary in batch_ref_summaries
                    ],
                    n_process=4,
                    disable=["parser", "ner"],
                ):
                    nrclex = copy.copy(nrc)
                    nrclex.load_raw_text(
                        " ".join([token.lemma_ for token in ref_summary])
                    )
                    batch_nrclex_ref_summaries.append(nrclex)

            else:
                for ref_summary in batch_ref_summaries:
                    nrclex = copy.copy(nrc)
                    nrclex.load_raw_text(ref_summary.lower())
                    batch_nrclex_ref_summaries.append(nrclex)

            # Generar la cadena de emociones a partir de los nrclex_summaries
            batch_emotion_chains = [
                get_emotion_chain(nrclex_summary)
                for nrclex_summary in batch_nrclex_ref_summaries
            ]

            sep_emo = "[emotions]"
            sep_summ = "[summary]"
            for i in range(len(batch_ref_summaries)):
                sample = {
                    "document": batch_documents[i],
                    "summary": "%s %s %s %s"
                    % (
                        sep_emo,
                        batch_emotion_chains[i],
                        sep_summ,
                        batch_ref_summaries[i],
                    ),
                }
                fw.write(json.dumps(sample) + "\n")
