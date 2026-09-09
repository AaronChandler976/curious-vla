# VLM-Guided Curiosity for Manipulation Skill Discovery

**Team:** Aaron Chandler — CMPE249, Intelligent Autonomous Systems, Fall 2026
**Selected track:** Research (VLA / robotics lane)

## Abstract

Vision-Language-Action (VLA) models for robotic manipulation are almost entirely trained on task-labeled teleoperated demonstrations, which are expensive to collect and discard any interaction data that lacks an explicit task instruction. Developmental robotics research argues that biological agents instead bootstrap "how to move" through unsupervised, curiosity-driven exploration — motor babbling — well before they learn "what to do." A closely related paper (World-Env) has already shown that autonomously-collected exploration data can measurably improve VLA fine-tuning on LIBERO — so this project does not set out to re-demonstrate that broad claim. Instead, it asks a narrower, still-open question: **does the *source* of that exploration data matter?** Specifically, do curiosity-driven (self-supervised prediction error) or VLM-guided semantic exploration policies — neither of which requires an already-competent policy to perturb, unlike prior work — produce exploration data that helps downstream VLA fine-tuning *more* than either undirected random play or a perturbed-competent-policy baseline?

This project tests that question in simulation (LIBERO) using an open small VLA (SmolVLA), with no physical hardware required.

## Repository Structure

- `docs/literature-survey.md` — SOTA survey (10 papers/models, 2023–2026)
- `docs/proposal.md` — problem formulation and technical approach
- `docs/ai-novelty-feasibility-audit.md` — AI-assisted novelty/red-ocean critique
- `src/` — exploration policy, intrinsic reward module, fine-tuning/evaluation scripts (to be added)
- `experiments/` — logs and results (to be added)

## Status

Proposal stage — topic selected 2026-09-08 via an AI-assisted topic-scouting process (landscape search → gap identification → novelty/feasibility audit). See `docs/ai-novelty-feasibility-audit.md` for the critique this selection was checked against.
