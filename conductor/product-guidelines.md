# Product Guidelines: Tiny MLP Permutation Invariance

## Overview
These guidelines ensure that all project communications, documentation, and user interfaces maintain a consistent, professional, and highly usable standard. The focus is on clarity, efficiency, and a seamless developer experience.

## Communication & Prose Style
- **Concise and Action-Oriented**: All documentation and internal communication should prioritize brevity and direct action. Use clear, imperative language (e.g., "Run this script," "Check the output") to guide the user through the process.
- **Precision**: While being concise, ensure that technical terms and procedures are described with absolute accuracy. Avoid ambiguity.

## Branding & Tone
- **Minimalist and Professional**: Maintain a clean, distraction-free tone in all project artifacts. The focus should be entirely on the technical content and the empirical results.
- **Reliability**: Use a tone that conveys confidence in the technical logic and the reproducibility of the results.

## UX & UI Principles (CLI-First)
- **CLI-First Design**: The primary interaction with the project is through the command-line interface. Each script must have a well-defined set of arguments and clear, predictable behavior.
- **Clean and Informative Outputs**: Every command should produce human-readable output that clearly indicates the current status (e.g., "Loading weights," "Applying permutation") and the final result (e.g., "PASS," "FAIL").
- **Visual Feedback**: Use appropriate formatting (e.g., bolding, color coding if supported) to highlight critical information like success/failure and key metrics (max difference).

## Documentation Standards
- **README Driven**: The `README.md` should serve as the central source of truth for setup and usage.
- **Inline Documentation**: Use clear, concise comments within the Python scripts to explain the "why" behind specific weight transformations and verification steps.
