# Blindfold SDK Demo

Interactive notebook showcasing PII detection, tokenization, redaction, masking, hashing, and synthesis — all running **locally** with zero API keys required.

## Run on Hugging Face

<!-- TODO: Add HF Spaces link once published -->

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook blindfold_demo.ipynb
```

## What's inside

| Section | Description |
|---------|-------------|
| Detect | Find PII with entity types, positions, and confidence scores |
| Tokenize & Detokenize | Reversible PII replacement for safe LLM calls |
| Redact | Permanent PII removal |
| Mask | Partial hiding (show first/last N chars) |
| Hash | Deterministic hashing for analytics |
| Synthesize | Realistic fake data replacement |
| Policies | Compare `basic`, `gdpr_eu`, `hipaa_us`, `pci_dss`, `strict` |
| Multi-Locale | US, EU, and international PII formats |
| Batch Processing | Process multiple texts at once |
| LLM Integration | Full tokenize-LLM-detokenize flow (simulated) |
| Performance | Benchmark vs Presidio |
