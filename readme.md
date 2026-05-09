# Symphonic Flow: AI Music Generation & The Alignment Problem

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Transformer](https://img.shields.io/badge/Architecture-Transformer-success)

MIDI file drive link: https://drive.google.com/drive/folders/1hKezbqXwlwtFiFyXrqY0JxP0rMSSisQ6?usp=drive_link

Symphonic Flow is a deep learning research project exploring the algorithmic generation of MIDI-based music. This repository documents the architectural progression from basic sequence memorizers to a highly controlled, genre-conditioned Autoregressive Transformer. 

Crucially, this project also investigates the limits of alignment, demonstrating how Reinforcement Learning from Human Feedback (RLHF) can lead to catastrophic "reward hacking" when applied to creative tasks with limited human data.

## 🧠 Architectural Progression

The project is divided into four distinct phases, tracking the evolution of the model's complexity and musical understanding.

### Task 1: The Baseline Autoencoder (LSTM)
An LSTM-based Encoder-Decoder model trained to compress and reconstruct MIDI sequences. 
* **Result:** Achieved low mathematical loss (`0.11 MSE`) but functioned primarily as a memorizer. Due to the nature of Mean Squared Error, the model struggled with dissonant "midpoint" predictions and failed to maintain long-term rhythmic structure.

### Task 2: Multi-Genre VAE
A Variational Autoencoder designed to map different musical styles into an organized latent space.
* **Result:** Introduced moderate genre control but still struggled with the "awkward silences" and repetition inherent to standard RNN-based generation.

### Task 3: The Champion Model (Genre-Conditioned Transformer)
A custom Transformer architecture using autoregressive training, dual-embedding integration (combining Token IDs with Genre IDs), and sinusoidal positional encoding.
* **Result:** Successfully learned the "grammar" of music, achieving strong rhythm diversity, low repetition, and a highly stable validation perplexity of `5.31`.

### Task 4: The RLHF Failure (A Cautionary Tale)
An attempt to align the Task 3 Transformer with human preferences using the REINFORCE algorithm and a small dataset of human-rated tracks.
* **Result:** A prime example of **Mode Collapse / Reward Hacking**. The AI bypassed musical rules entirely to maximize the mathematical reward, resulting in chaotic noise, broken genre control, and a perplexity spike to `14.20`.

---

## 🎼 The "Musical Rulebook" (Logit Warping Heuristics)
To bridge the gap between raw statistical probability and actual music theory, the Task 3 Transformer utilizes a custom suite of generation heuristics applied during the sampling phase:

* **Tonal Biasing:** Applies a `+1.2` logit boost to notes within the C-Major/A-Minor scale and a heavy penalty to accidentals, ensuring melodic safety.
* **Reverse Momentum:** A decaying penalty array (multiplying by `0.85` per step) that heavily penalizes recently played notes to force melodic exploration and prevent infinite looping.
* **The Silence Nuke:** A massive `-8.0` penalty applied to "TimeShift" tokens if the model attempts to generate extended, unnatural silences.
* **Octave Fatigue & 3-Octave Cage:** Prevents the AI from lingering in a single high/low register for too long and restricts impossible physical jumps (greater than 18 semitones) across the keyboard.

---

## 📊 Performance Metrics

| Model | Perplexity | Rhythm Diversity (↑) | Repetition Ratio (↓) | Human Score (out of 5) | Genre Control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random Generator | - | 0.150 | 0.000 | 1.00 | None |
| Markov Chain | - | 0.017 | 0.270 | 1.00 | Weak |
| Task 1: Autoencoder | - | NaN | NaN | 1.50 | Single |
| Task 2: VAE | - | 0.003 | 0.162 | 3.25 | Moderate |
| **Task 3: Transformer** | **5.31** | **0.215** | **0.006** | **4.75** | **Strong** |
| Task 4: RLHF | 14.20 | 0.046 | 0.019 | 1.50 | Broken |

> *Note: Rhythm Diversity and Repetition Ratio for Task 1 returned NaN due to the model's inability to generate continuous, measurable sequences without breaking into silence.*

---

## ⚙️ Installation & Usage

### Dependencies
Ensure you have Python 3.8+ installed. Install the required packages:
```bash
pip install -r requirements.txt
