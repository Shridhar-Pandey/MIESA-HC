# MIESA-HC

# MIESA-HC: Meeting Intent Evidence Scoring Algorithm with Hybrid Classification

## Overview

**MIESA-HC** stands for **Meeting Intent Evidence Scoring Algorithm with Hybrid Classification**.  
It is a lightweight and interpretable meeting-intelligence framework designed to convert raw meeting transcripts or audio-based meeting recordings into structured, actionable outputs.

Traditional meeting summarization systems mainly generate a short summary of the meeting. However, a summary alone does not clearly answer practical questions such as:

- Who is responsible for the next task?
- What decision was finally taken?
- What is blocking the progress?
- Which questions are still open?
- What should be included in the final meeting minutes?

MIESA-HC addresses this gap by performing **sentence-level meeting intent classification** and post-processing. Each sentence in a meeting transcript is classified into one of five meeting-specific categories:

1. **Action**
2. **Decision**
3. **Blocker**
4. **Question**
5. **Discussion**

Along with classification, the system also extracts:

- Assignee information
- Deadline cues
- Priority level
- Action items
- Decisions
- Blockers
- Open questions
- Structured meeting minutes

The framework is designed to be computationally efficient, reproducible in Google Colab, and usable without depending fully on large pretrained language models.

---

## Project Title

**MIESA-HC: A Weakly Supervised Hybrid Framework for Meeting Intelligence**

---

## Problem Statement

Meetings generate a large amount of valuable information, but most of this information remains in unstructured form. In academic, organizational, and professional meetings, important points such as decisions, action items, blockers, and open questions are often hidden inside long transcripts.

After a meeting, someone usually has to manually read the transcript and prepare meeting minutes. This process is:

- Time-consuming
- Error-prone
- Inconsistent
- Difficult to scale
- Dependent on human attention

Existing meeting-processing systems mostly focus on summarization. While summarization is useful, it may not clearly identify actionable meeting outcomes. Therefore, there is a need for a lightweight and interpretable system that can extract structured meeting intelligence from transcripts.

---

## Aim of the Project

The main aim of MIESA-HC is to develop a meeting-intelligence system that can automatically analyze meeting transcripts and generate structured outputs such as action items, decisions, blockers, open questions, and meeting minutes.

---

## Objectives

The major objectives of this project are:

- To classify meeting sentences into meaningful intent categories.
- To reduce manual effort required in preparing meeting minutes.
- To identify actionable tasks from meeting transcripts.
- To detect decisions taken during a meeting.
- To identify blockers or risks mentioned during discussions.
- To extract open questions that require follow-up.
- To infer assignees, deadlines, and priority levels.
- To design a lightweight model that does not require heavy pretrained transformer models.
- To provide a practical Gradio-based interface for text and audio-based meeting inputs.

---

## Key Features

- Sentence-level meeting intent classification
- Weak supervision-based label generation
- Rule-based label refinement
- Human-corrected gold-label integration
- Handcrafted feature engineering
- Weighted softmax classifier implemented using NumPy
- Text-based meeting transcript analysis
- Audio-based meeting analysis using Whisper transcription
- Optional speaker diarization support
- Gradio web interface
- Monitoring dashboard for meeting statistics
- Meeting minutes generation
- Exportable JSON and text outputs

---

## Meeting Intent Classes

MIESA-HC classifies each sentence into one of the following five classes:

| Class | Meaning | Example |
|---|---|---|
| Action | A task or activity that needs to be done | "Rahul will submit the final report by Friday." |
| Decision | A final agreement or conclusion | "We agreed to use Google Colab for model training." |
| Blocker | A problem, risk, or dependency | "The deployment is blocked due to authentication failure." |
| Question | A query or unresolved issue | "Can we complete the testing before Monday?" |
| Discussion | General conversation or background information | "The team reviewed the current project status." |

---

## Theoretical Background

### 1. Meeting Intelligence

Meeting intelligence refers to the process of automatically extracting useful and structured information from meeting conversations. Unlike simple summarization, meeting intelligence focuses on identifying practical meeting outcomes.

A meeting-intelligence system should be able to answer:

- What was discussed?
- What was decided?
- What tasks were assigned?
- Who is responsible?
- What deadlines were mentioned?
- What risks or blockers were raised?
- What questions remain unanswered?

MIESA-HC is built around this idea. Instead of only generating a paragraph summary, it creates structured meeting outputs that are directly useful for follow-up work.

---

### 2. Weak Supervision

Weak supervision is a machine learning approach where training labels are generated using rules, heuristics, patterns, or programmatic logic instead of full manual annotation.

In this project, sentence-level gold labels are not directly available in the MeetingBank dataset. Therefore, the system first creates weak labels automatically using evidence scores.

For example:

- Sentences with words like "will", "assign", "complete", or "submit" may indicate an **Action**.
- Sentences with phrases like "we decided", "approved", or "agreed" may indicate a **Decision**.
- Sentences with words like "blocked", "issue", "delay", or "problem" may indicate a **Blocker**.
- Sentences ending with a question mark may indicate a **Question**.
- Sentences that do not strongly match other classes may be treated as **Discussion**.

Weak supervision reduces the need for large manually annotated datasets and makes the project more practical in low-resource academic settings.

---

### 3. Hybrid Classification

The term **Hybrid Classification** in MIESA-HC means that the final system does not rely on only one technique. Instead, it combines multiple components:

1. **Handcrafted feature extraction**
2. **Weak evidence scoring**
3. **Rule-based label refinement**
4. **Human-corrected gold-label integration**
5. **Weighted softmax classification**

This hybrid design improves interpretability and allows the system to work without depending completely on large pretrained transformer models.

---

## Proposed Methodology

The complete MIESA-HC pipeline follows six major stages:

```text
Input Meeting Transcript / Audio
              |
              v
       Preprocessing
              |
              v
     Feature Extraction
              |
              v
   Weak Evidence Scoring
              |
              v
    Rule-Based Refinement
              |
              v
   Hybrid Softmax Classifier
              |
              v
      Post-Processing
              |
              v
Structured Meeting Intelligence

```


