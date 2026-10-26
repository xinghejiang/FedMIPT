# FedMIPT

Federated Multi-Interest Prototype Transport for Multimodal Cross-Domain Recommendation.

## Repository Structure

```text
main.py                 Training and evaluation entry point
fedmipt/
  config.py             Configuration loading and validation
  data.py               Interaction and feature loading
  models.py             Local recommendation model
  interests.py          Multi-interest prototype construction
  transport.py          Null-aware optimal transport
  readout.py            Candidate-conditioned interest readout
  transfer.py           Cross-domain package exchange and alignment
  federated.py          Shared projector aggregation
  trainer.py            Training and model selection
  evaluation.py         Recommendation evaluation
  metrics.py            Ranking metrics
  checkpoint.py         Checkpoint saving and loading
  utils.py              General utilities
configs/                Model configuration files
data/                   Dataset placeholder
requirements.txt        Python dependencies
```
