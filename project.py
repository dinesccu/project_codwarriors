import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# DATASET PATH
# =============================
file_path = r"C:\Users\HP\Desktop\Conversational_Transcript_Dataset.json"

# =============================
# LOAD DATA
# =============================
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    data = list(data.values())[0]

# =============================
# MAIN DATAFRAME
# =============================
df_main = pd.DataFrame(data)

print("Columns Found:")
print(df_main.columns)

# =============================
# EXPLODE CONVERSATION
# =============================
df = df_main.explode("conversation").reset_index(drop=True)
conv_df = pd.json_normalize(df["conversation"])
df = pd.concat([df.drop(columns=["conversation"]), conv_df], axis=1)

# =============================
# CLEAN TEXT
# =============================
df["text"] = df["text"].astype(str).str.lower().str.strip()

# =============================
# SORT DATA
# =============================
if "transcript_id" in df.columns and "turn_id" in df.columns:
    df = df.sort_values(by=["transcript_id", "turn_id"])

# =============================
# GROUP CONVERSATIONS
# =============================
conversations = df.groupby("transcript_id")
print("\nTotal Conversations:", len(conversations))

# =============================
# EVENT COLUMN
# =============================
event_col = "intent"

# =============================
# LABEL UNDERSTANDING
# =============================
event_conversations = df[df[event_col].notna()]["transcript_id"].unique()
non_event_conversations = df[df[event_col].isna()]["transcript_id"].unique()

print("\nEvent Conversations:", len(event_conversations))
print("Non Event Conversations:", len(non_event_conversations))

# =====================================================
# STEP 4 — PATTERN ANALYSIS
# =====================================================
anger_words = ["angry", "frustrated", "worst", "bad", "not happy", "complaint"]
delay_words = ["waiting", "delay", "hold", "long time", "slow"]
agent_mistake_words = ["wrong", "mistake", "incorrect", "error", "not updated"]

event_phrases = []
anger_count = delay_count = mistake_count = 0

for conv_id in event_conversations:
    conv_group = conversations.get_group(conv_id)
    for _, row in conv_group.tail(3).iterrows():
        text = row["text"]
        event_phrases.append(text)
        if any(w in text for w in anger_words):
            anger_count += 1
        if any(w in text for w in delay_words):
            delay_count += 1
        if any(w in text for w in agent_mistake_words):
            mistake_count += 1

print("\nTop Phrases Before Event:")
for p, c in Counter(event_phrases).most_common(5):
    print(c, "→", p)

# =====================================================
# STEP 5 — CAUSAL REASONING
# =====================================================
total_events = len(event_conversations)
anger_score = anger_count / total_events
delay_score = delay_count / total_events
mistake_score = mistake_count / total_events

print("\nAnger Probability:", round(anger_score, 3))
print("Delay Probability:", round(delay_score, 3))
print("Agent Mistake Probability:", round(mistake_score, 3))

# =====================================================
# STEP 5.1 — EVENT PREDICTION
# =====================================================
def predict_event(text):
    if any(w in text for w in anger_words):
        return "Customer Frustration"
    elif any(w in text for w in delay_words):
        return "Long Wait"
    elif any(w in text for w in agent_mistake_words):
        return "Agent Error"
    else:
        return "none"

df["predicted_event"] = df["text"].apply(predict_event)

# =====================================================
# STEP A — AUTO INTENT → CAUSE MAPPING
# =====================================================
intent_to_cause = {}

for intent, group in df[df[event_col].notna()].groupby(event_col):
    valid_causes = group["predicted_event"]
    valid_causes = valid_causes[valid_causes != "none"]
    if not valid_causes.empty:
        intent_to_cause[intent] = valid_causes.value_counts().idxmax()

print("\nAUTO GENERATED INTENT → CAUSE MAP")
for k, v in intent_to_cause.items():
    print(f"{k} → {v}")

# =====================================================
# STEP B — TRUE CAUSE COLUMN
# =====================================================
df["true_cause"] = df[event_col].map(intent_to_cause)

# =====================================================
# STEP 5.2 — CORRECT ACCURACY
# =====================================================
valid_rows = df[(df["true_cause"].notna()) & (df["predicted_event"] != "none")]

print("\nSample True Cause vs Prediction:")
print(valid_rows[["true_cause", "predicted_event"]].head())

accuracy = accuracy_score(valid_rows["true_cause"], valid_rows["predicted_event"])
print("\n✅ Accuracy:", round(accuracy, 3))

print("\nClassification Report:")
print(classification_report(valid_rows["true_cause"], valid_rows["predicted_event"], zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(valid_rows["true_cause"], valid_rows["predicted_event"]))

# =====================================================
# 📊 GRAPHS
# =====================================================
plt.figure()
plt.bar(["Anger", "Delay", "Mistake"], [anger_count, delay_count, mistake_count])
plt.title("Pattern Signal Counts")
plt.show()

plt.figure()
plt.bar(["Anger", "Delay", "Mistake"], [anger_score, delay_score, mistake_score])
plt.title("Causal Probability Scores")
plt.show()

# =====================================================
# 🔎 STEP 6 — EVIDENCE EXTRACTION
# =====================================================
def extract_evidence(query, top_n=5):
    query = query.lower()
    evidence = df[df["text"].str.contains(query, na=False)]

    if evidence.empty:
        print("❌ No evidence found")
        return []

    print(f"\n✅ Found {len(evidence)} evidence rows")

    results = []
    for _, row in evidence.head(top_n).iterrows():
        record = {
            "transcript_id": row["transcript_id"],
            "turn_id": row.get("turn_id", "NA"),
            "speaker": row.get("speaker", "NA"),
            "intent": row.get(event_col, "NA"),
            "predicted_event": row["predicted_event"],
            "text": row["text"]
        }
        print(record)
        results.append(record)

    return results

# =====================================================
# 🧠 STEP 7 — INTERACTIVE QUERY SYSTEM
# =====================================================
def query_system():
    print("\n💬 Evidence Query System")
    print("Examples: delay, angry, wrong, complaint")
    print("Type 'exit' to stop")

    while True:
        q = input("\nAsk Query: ").lower()
        if q == "exit":
            break
        extract_evidence(q)

# RUN
query_system()