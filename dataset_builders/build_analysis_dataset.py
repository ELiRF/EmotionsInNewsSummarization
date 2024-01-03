import copy
import pickle as pkl
from functools import partial
from random import shuffle

import numpy as np
import spacy
from datasets import load_dataset, load_metric
from generation_hyperparameters import generation_hyperparameters
from nltk import sent_tokenize
from NRCLex.nrclex import NRCLex
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ExtractiveBaselines:
    @staticmethod
    def lead(inputs, dataset_name):
        k = 1 if dataset_name == "xsum" else 3
        return [
            "\n".join(sent_tokenize(document)[:k])
            for document in inputs["batch_documents"]
        ]

    @staticmethod
    def random(inputs, dataset_name):
        k = 1 if dataset_name == "xsum" else 3
        batch_gen_summaries = []
        for document in inputs["batch_documents"]:
            sents = sent_tokenize(document)
            shuffle(sents)
            rnd_sents = "\n".join(sents[:k])
            batch_gen_summaries.append(rnd_sents)
        return batch_gen_summaries

    @staticmethod
    def extractive_oracle(inputs, dataset_name, rouge_metric):
        batch_documents = inputs["batch_documents"]
        batch_summaries = inputs["batch_summaries"]
        batch_gen_summaries = []
        for i in range(len(batch_documents)):
            document = batch_documents[i]
            ref_summary = batch_summaries[i]
            gen_summary_sents = []
            document_sents = sent_tokenize(document)
            ref_summary_sents = sent_tokenize(ref_summary)
            for summary_sent in ref_summary_sents:
                if len(document_sents) == 0:
                    gen_summary_sents = []
                    continue
                rouge_scores = rouge_metric.compute(
                    predictions=[summary_sent for _ in document_sents],
                    references=document_sents,
                    use_stemmer=True,
                    use_agregator=False,
                )
                rouge_scores = np.array(
                    [
                        (
                            rouge_scores["rouge1"][i].fmeasure
                            + rouge_scores["rouge2"][i].fmeasure
                            + rouge_scores["rougeLsum"][i].fmeasure
                        )
                        / 3.0
                        for i in range(len(rouge_scores["rouge1"]))
                    ]
                )
                idx_most_sim_sent = rouge_scores.argmax()
                most_sim_sent = document_sents[idx_most_sim_sent]
                gen_summary_sents.append(most_sim_sent)
            batch_gen_summaries.append("\n".join(gen_summary_sents))
        return batch_gen_summaries


def prefix_allowed_tokens_fn(batch_id, input_ids, tok_prefixes, tokenizer):
    position = len(input_ids)
    if position < len(tok_prefixes["input_ids"][batch_id]):
        if (
            tok_prefixes["input_ids"][batch_id][position - 1]
            in tokenizer.all_special_ids
        ):
            return None
        return (
            tok_prefixes["input_ids"][batch_id][position - 1].view(1).tolist()
        )
    return None


def get_emotion_chain(sample_nrclex):
    emotion_chain = []
    for sentence in sample_nrclex.sentences:
        emotions_list = []
        for word in sample_nrclex.affect_dict:
            if word in sentence:
                emotions_list.append(word)

        emotion_chain.append(" | ".join(emotions_list))

    return " ||| ".join(emotion_chain)


def get_prefixes_for_oracle(nrc, batch_ref_summaries, tokenizer):
    emotion_chains = []
    for ref_summary in batch_ref_summaries:
        summary_nrclex = copy.copy(nrc)
        summary_nrclex.load_raw_text(
            " ".join([token.lemma_ for token in nlp(ref_summary.lower())])
        )
        emotion_chains.append(get_emotion_chain(summary_nrclex))

    tok_prefixes = tokenizer(
        emotion_chains,
        max_length=256,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    return tok_prefixes


dataset_name = "xsum"
doc_summ_keys = {
    "document": "document",
    "summary": "summary",
}

"""
Available models:

(Extractive)
"lead"
"random"
"extractive_oracle

(Abstractive)
"facebook/bart-large-cnn"
"google/pegasus-cnn_dailymail"
"t5-base"
"facebook/bart-large-xsum"
"facebook/pegasus-xsum"
"./checkpoints/bart-JES-cnn_dailymail"
"./checkpoints/pegasus-JES-cnn_dailymail"
"./checkpoints/bart-JES-xsum"
"./checkpoints/pegasus-JES-xsum"
# abstractive_oracle puede ser True solo para los modelos JES
"""

model_name = "t5-base"  # "./checkpoints/bart-JES-xsum"
abstractive_oracle = False  # True #True

if abstractive_oracle:
    assert "-JES-" in model_name

version = None  # "3.0.0"  # None
batch_size = 32  # 32  # 16
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
ts = dataset["test"]
documents = [doc for doc in ts[doc_summ_keys["document"]]]
ref_summaries = [ref_summ for ref_summ in ts[doc_summ_keys["summary"]]]


if model_name not in ["lead", "random", "extractive_oracle"]:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "t5" in model_name:
        inputs = tokenizer(
            ["summarize: " + document for document in documents],
            max_length=512,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to("cuda")
    else:
        inputs = tokenizer(
            documents,
            max_length=512,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to("cuda")
else:
    if model_name == "extractive_oracle":
        model = partial(
            getattr(ExtractiveBaselines, model_name),
            rouge_metric=load_metric("rouge"),
        )
    else:
        model = getattr(ExtractiveBaselines, model_name)

rouge_metric = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

output_dict = {
    "documents": [],
    "ref_summaries": [],
    "gen_summaries": [],
    "nrclex_documents": [],
    "nrclex_ref_summaries": [],
    "nrclex_gen_summaries": [],
    "rouge_scores": {"rouge1_f1": [], "rouge2_f1": [], "rougeLsum_f1": []},
    "bert_scores": {"f1": []},
}

print("Model:", model_name)
gen_args = generation_hyperparameters[model_name][dataset_name]
gen_args["early_stopping"] = True
print("Dataset:", dataset_name)
print(
    "Generation hyper-params of the model:",
    gen_args,
)
print("Is abstractive oracle?:", abstractive_oracle)

for i in range(0, len(documents), batch_size):
    print("Samples processed:", i)

    batch_documents = documents[i : i + batch_size]
    batch_ref_summaries = ref_summaries[i : i + batch_size]

    if model_name not in ["lead", "random", "extractive_oracle"]:
        batch_input_ids = inputs["input_ids"][i : i + batch_size]
        if abstractive_oracle:
            tok_prefixes = get_prefixes_for_oracle(
                nrc, batch_ref_summaries, tokenizer
            )

            gen_args["prefix_allowed_tokens_fn"] = partial(
                prefix_allowed_tokens_fn,
                tok_prefixes=tok_prefixes,
                tokenizer=tokenizer,
            )
            batch_gen_ids = model.generate(batch_input_ids, **gen_args)

        else:
            batch_gen_ids = model.generate(batch_input_ids, **gen_args)

        batch_gen_summaries = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            .replace(". ", " .\n")
            .replace("<n>", "\n")
            for g in batch_gen_ids
        ]

        if "-JES-" in model_name:
            summ_marker = "[summary]"
            for i, summary in enumerate(batch_gen_summaries):
                pos_start = summary.find(summ_marker)
                pos_start = 0 if pos_start == -1 else pos_start
                batch_gen_summaries[i] = batch_gen_summaries[i][
                    pos_start + len(summ_marker) :
                ].strip()

    else:
        batch_gen_summaries = model(
            {
                "batch_documents": batch_documents,
                "batch_summaries": batch_ref_summaries,
            },
            dataset_name,
        )

    batch_rouge_scores = rouge_metric.compute(
        predictions=batch_gen_summaries,
        references=batch_ref_summaries,
        use_stemmer=True,
        use_agregator=False,
    )

    batch_bert_scores = bertscore_metric._compute(
        predictions=batch_gen_summaries,
        references=batch_ref_summaries,
        lang="en",
        rescale_with_baseline=True,
    )

    if lemmatize_nrc:
        batch_nrclex_documents = []
        batch_nrclex_ref_summaries = []
        batch_nrclex_gen_summaries = []

        for document in batch_documents:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(
                " ".join([token.lemma_ for token in nlp(document.lower())])
            )
            batch_nrclex_documents.append(nrclex)

        for ref_summary in batch_ref_summaries:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(
                " ".join([token.lemma_ for token in nlp(ref_summary.lower())])
            )
            batch_nrclex_ref_summaries.append(nrclex)

        for gen_summary in batch_gen_summaries:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(
                " ".join([token.lemma_ for token in nlp(gen_summary.lower())])
            )
            batch_nrclex_gen_summaries.append(nrclex)

    else:
        for document in batch_documents:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(document.lower())
            batch_nrclex_documents.append(nrclex)

        for ref_summary in batch_ref_summaries:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(ref_summary.lower())
            batch_nrclex_ref_summaries.append(nrclex)

        for gen_summary in batch_gen_summaries:
            nrclex = copy.copy(nrc)
            nrclex.load_raw_text(gen_summary.lower())
            batch_nrclex_gen_summaries.append(nrclex)

    output_dict["documents"] += batch_documents
    output_dict["ref_summaries"] += batch_ref_summaries
    output_dict["gen_summaries"] += batch_gen_summaries

    output_dict["rouge_scores"]["rouge1_f1"] += [
        score.fmeasure for score in batch_rouge_scores["rouge1"]
    ]
    output_dict["rouge_scores"]["rouge2_f1"] += [
        score.fmeasure for score in batch_rouge_scores["rouge2"]
    ]
    output_dict["rouge_scores"]["rougeLsum_f1"] += [
        score.fmeasure for score in batch_rouge_scores["rougeLsum"]
    ]

    output_dict["bert_scores"]["f1"] += batch_bert_scores["f1"]

    output_dict["nrclex_documents"] += batch_nrclex_documents
    output_dict["nrclex_ref_summaries"] += batch_nrclex_ref_summaries
    output_dict["nrclex_gen_summaries"] += batch_nrclex_gen_summaries


fname = "%s+%s" % (model_name.split("/")[-1], dataset_name.split("/")[-1])
if lemmatize_nrc:
    fname += "+lemmatized"
if abstractive_oracle:
    fname += "+oracle"

with open("%s.pkl" % fname, "wb") as fw:
    pkl.dump(output_dict, fw)
