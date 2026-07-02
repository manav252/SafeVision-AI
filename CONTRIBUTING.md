# Contributing to SafeVision AI

Thank you for improving SafeVision AI. This project is an industrial safety intelligence platform combining CCTV analytics, plant context, risk scoring, and response workflows.

## Development Workflow

1. Fork or clone the repository.
2. Create a feature branch.
3. Install dependencies with `pip install -r requirements.txt`.
4. Keep changes scoped and documented.
5. Run `python -m compileall SafeVision-AI` before opening a pull request.

## Code Standards

- Use clear names and small functions.
- Keep safety logic explainable.
- Avoid committing private footage, credentials, or plant-specific data.
- Keep sample data synthetic or licensed for demo use.

## Security

Do not commit `.env`, production keys, CCTV credentials, or real worker identity data.

