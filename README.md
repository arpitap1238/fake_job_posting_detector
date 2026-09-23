# fake_job_posting_detector
# Fake Job & Internship Posting Detector

I built this after noticing how many sketchy internship postings show up on job boards and LinkedIn — vague descriptions, no company info, "work from home, no experience needed, apply immediately" type listings that are obviously designed to catch people who are eager and not paying close attention. Students are usually the ones who fall for these, since we're the ones desperate enough to apply to almost anything.

So instead of just being annoyed by it, I built a tool that takes a job/internship posting and tells you how likely it is to be fake, along with the actual reasons why — not just a score with no explanation.

## How it works

The model is trained on ~17,880 real job postings (about 800 of which are labeled fraudulent), using a public dataset originally built for exactly this kind of research. Each posting gets converted into two types of features: the actual words in the description (using TF-IDF), and a set of "red flags" I engineered myself based on patterns I found during exploratory analysis — things like whether a salary is mentioned, whether the posting has any company background info, and whether it uses phrases like "work from home" or "no experience needed."

One thing worth mentioning: only about 5% of postings in the dataset are actually fake, so a model that just guessed "real" every time would already be 95% "accurate" while being completely useless. Because of that, I used F1-score instead of accuracy to actually judge performance, and used SMOTE to balance the training data (without touching the test set, so evaluation stays realistic).

I trained and compared three models — Logistic Regression, Random Forest, and XGBoost — and Random Forest came out ahead:

| Model               | Precision (Fake) | Recall (Fake) | F1-Score |
|---------------------|------------------|---------------|----------|
| Logistic Regression | 0.28             | 0.77          | 0.41     |
| Random Forest       | 0.93             | 0.65          | 0.77     |
| XGBoost             | 0.88             | 0.65          | 0.74     |

I also adjusted the default classification threshold from 0.5 down to 0.4, since missing a real scam felt worse than occasionally flagging a legitimate posting for a second look. That pushed recall from 65% to 76%, with F1 improving slightly to 0.78.

For comparison, a similar project using this same dataset (Sharma, 2024) got F1-scores of around 0.22–0.23 using text features alone — this project does noticeably better, mostly because of the structural red-flag features rather than the text itself.

## Some things I found along the way

Not everything I assumed going in turned out to be true, which was honestly the most interesting part of this project:

- Whether a posting has a company logo/profile turned out to be one of the strongest signals — real postings had one 82–84% of the time, fake ones only about 32%.
- I expected fake postings to avoid mentioning salary, since scammers wouldn't want to commit to a number. The opposite was true — fake postings mentioned salary more often (25.8% vs 15.5%), probably because a specific number makes the listing more enticing.
- Just counting common words didn't separate fake from real postings well — both use pretty normal-sounding language. Specific phrases like "work from home" and "no experience needed" were far more telling than individual words.

## Explainability

Every prediction comes with a breakdown of what actually drove it, using SHAP — so instead of just "73% fake," the app shows you which specific factors pushed the decision one way or the other. That felt important for something people are actually going to use to make a decision about a real posting.

## A limitation worth being upfront about

The deployed app lets you paste in raw text (copied straight from LinkedIn or an email), but a few features the model was trained on — like whether the posting has a company logo — can't be detected from plain text at all. Those default to "not present," which can slightly bias the model toward predicting "fake" even for genuinely real postings with thin descriptions. It's a real trade-off of adapting a model trained on structured data to raw, unstructured input, and something I'd want to address with a text-only model variant if I extend this further.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, SHAP, Streamlit

## Live demo
https://fakejobpostingdetector-dru6zdby4rji2yjd4tvkxt.streamlit.app/

## What I'd add next

- Extracting job titles from postings and recommending similar verified listings
- An API version so this could plug into other tools
- A separate model trained specifically for text-only input, to fix the logo-detection limitation above
