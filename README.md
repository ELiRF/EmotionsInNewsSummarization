# Abstractive Summarizers Get Emotional on News Summarization 😊😨😠

This repository contains the code to reproduce the experiments from "Abstractive Summarizers Get Emotional on News Summarization" paper, along with additional experiments aimed to condition summarization models for emotion-controllable generation.

## 📁 Scripts
The scripts in **dataset_builders/** can be used to store experiment files:
- **build_analysis_dataset.py**: builds and stores the dictionaries for the analysis of the models.
- **build_modeling_dataset.py**: builds and stores the datasets for training the emotion-conditioned models.
- **build_nrclex_dataset.py**: builds and stores the NRCLex information in a dataset (useful for calculating emotion statistics in the corpora).

The **analysis.ipynb** notebook computes the results of all the experiments by using the files generated with **dataset_builders/** scripts.

## 🔧 How to install
You need to install the base requirements:

```
pip install -r requirements.txt
```

or the development ones if you are planning to work in development mode:

```
pip install -r dev-requirements.txt
```

## 📖 Citation
---
```
@misc{emotional_summarization,
      title={{A}bstractive {S}ummarizers {G}et {E}motional on {N}ews {S}ummarization}, 
      author={Ahuir, Vicent and González, José-Ángel and Hurtado, Lluís-F. and Segarra, Encarna},
      year={2024},
      eprint={TBD},
}
```
## 🤝 Contribute
---

Feel free to contribute by raising an issue or doing a pull-request.