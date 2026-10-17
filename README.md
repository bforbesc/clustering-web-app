# 🧩 Clustering Web App

> **Learning case study** · Unsupervised machine learning · Preprocessing · Data visualization
>
> An experiment preserved for teaching. You can learn from the explanation, sample, and code even if the original app no longer runs.

[📚 Learning hub](https://github.com/bforbesc/Data-AI-with-Forbes) · [🧭 Beginner guide](https://github.com/bforbesc/Data-AI-with-Forbes/blob/main/START_HERE.md) · [🌐 Build and deploy a data app](https://github.com/bforbesc/Data-AI-with-Forbes/tree/main/templates/streamlit-starter)

## The idea before the tools

**What does it mean to find similar observations without supplying the answers?**

**Measurements → comparable scales → chosen number of groups → visual check**

This is a study of an approach, rather than a recommendation to adopt an old app. First explain what happens in each step; only then look at the library or framework that implements it.

| Tool in this project | What it does in the example |
|---|---|
| **Pandas** | Reads and inspects the table. |
| **scikit-learn (`sklearn`)** | Standardizes columns, fits KMeans, measures clusters, and computes a t-SNE view. |
| **Plotly** | Draws interactive graphs. |
| **Streamlit** | Wraps the experiment in a browser interface. |

This is **unsupervised learning**: no answer labels are supplied. In **supervised learning**, examples include answers used to train a predictor. Compare it with [the first-classifier lesson](https://github.com/bforbesc/Data-AI-with-Forbes/blob/main/notebooks/3-first-classifier.ipynb).

**Can a computer find groups in a table without being told the answers?**

This app groups similar observations and draws a map of the result. You can explore it without writing code. It is an experiment in **unsupervised learning**: learning from patterns without supplied answer labels.

Part of [Data & AI with Forbes](https://github.com/bforbesc/Data-AI-with-Forbes). This is an older Streamlit project, kept as a teaching reference; its pinned dependencies may need a compatible older Python environment.

## What it does, and why

You give the app a table. **KMeans** divides the rows into a number of groups you choose. **t-SNE** makes a two-dimensional picture so you can inspect the result. The picture helps you explore; it does not prove that the groups are meaningful in the real world.

A useful example is grouping shop visits by how often someone visits and how much they spend. The point is to ask what the groups might mean, then check the evidence before giving them labels.

## Start here: no coding

Use the [original hosted app](https://clustering-web-app.streamlit.app), if it is available, or ask your teacher to run it. Its availability is not guaranteed. If the app fails, use the discussion and worksheet below.

1. Open [the fictional shop-visits sample](examples/shop_visits.csv) on GitHub and choose **Download raw file** near the top right. Keep the `.csv` extension. CSV is simply a table saved as text with commas between values.
2. In the app, use **Upload your cleaned CSV file** to choose that downloaded file. Keep the separator set to a comma.
3. Tick **Show data**. There are **90 rows** and **2 columns**: `weekly_visits` and `monthly_spend_eur`. Each row is a fictional customer; no real personal data is used.
4. Find **Preferred number of clusters** on the left. Try **2**, **3**, then **4**, waiting for each result.
5. Tick **Show clusters description** and compare the average visits and spending in each group. Explain one difference in ordinary language.

The data was made with three deliberately separated patterns: occasional/lower-spend, regular/middle-spend, and frequent/higher-spend visits. Three groups should be a useful result to discuss, but cluster numbers and the picture's orientation are not fixed labels. Real datasets rarely have such tidy groups.

## Your first investigation

| Try | Record | Ask |
|---|---|---|
| Choose 2 clusters | A description of each group | Which different patterns have been combined? |
| Choose 3 clusters | Average visits and spending for each group | Can you describe them without calling one group better? |
| Choose 4 clusters | Which pattern appears to split | Does an extra group add useful information? |

**A good answer:** “With three clusters, I can describe customers with different visiting and spending patterns. That does not explain why they behave that way.”

**A conclusion to question:** “The computer found three types of people, so those types must be real.”

## What do the extra graphs mean?

You can do the activity without memorizing these terms:

| Label in the app | What to look for |
|---|---|
| SSE / elbow | Adding groups reduces within-group error. Look for a point where the improvement slows down. |
| Silhouette coefficient | Higher values suggest rows fit their own group more clearly than other groups. Values range from -1 to 1. |
| Davies–Bouldin index | Lower values suggest more separated, compact groups. It is non-negative and can exceed 1. |
| t-SNE | A map for inspecting nearby observations. Its axes are not visits or euros, and large-scale distances can mislead. |

The app standardizes the columns before clustering so their scales are more comparable. Otherwise a large range of euro values could dominate a small range of visit counts. The metrics inform a choice; they do not choose the business meaning for you.

## If the app is unavailable

On paper, group these six fictional observations: `(1 visit, €20)`, `(2, €25)`, `(6, €70)`, `(7, €80)`, `(11, €150)`, `(12, €160)`. First make two groups, then three. Compare your choices with a classmate's. Do you agree because of the data, or because you were asked for that number of groups?

## For teachers and optional local setup

Allow about 20 minutes. Use the bundled sample before trying another file. The current app tries cluster counts from 2 to 15 and uses t-SNE perplexity 30, so keep more than 30 rows. Avoid missing values, duplicates, text columns, and identifier columns for this first activity. Do not use real student or customer records.

The requirements are from an older Python stack. A compatible Python 3.10 environment is a starting point for the pinned versions, not a promise that every modern machine can install them. Do not ask beginners to debug dependencies as their first activity.

Download and extract the repository using **Code → Download ZIP**. In a terminal in that folder:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run ml-web-app.py
```

Use `python3` to create the environment if that is your Python command. Open the local address printed by Streamlit; stop with **Ctrl+C**. Older dependencies or the deployment may need maintenance before class.

## Optional: look inside

[ml-web-app.py](ml-web-app.py) reads the CSV, prepares the columns, standardizes the values, fits KMeans, and builds the graphs. Try finding `n_clusters`: it connects a visible slider to a model setting.

## Credits and reuse

Inspired by [Data Professor](https://github.com/dataprofessor) and his [data science web-app article](https://towardsdatascience.com/how-to-build-a-data-science-web-app-in-python-penguin-classifier-2f101ac389f3). Code and the fictional teaching sample use the existing [MIT License](LICENSE). [Questions or broken links](https://github.com/bforbesc/clustering-web-app/issues) are welcome.
