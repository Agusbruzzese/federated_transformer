# Spintronic Federated Learning System

TMR sensor array — EMD/HHT feature extraction + contrastive FL via Flower + FedProx

---

## Project structure

```
spintronic_fl/
├── docker-compose.yml          ← orchestrates 1 server + 15 clients
├── shared/
│   └── model.py                ← SpintronicEncoder + NTXentLoss (shared)
├── server/
│   ├── Dockerfile
│   ├── server.py               ← Flower FedProx server
│   ├── pretrain_server.py      ← pre-train encoder on full Zenodo features
│   └── requirements.txt
└── client/
    ├── Dockerfile
    ├── client.py               ← Flower NumPyClient
    ├── pipeline.py             ← EMD/HHT feature extraction from BSD files
    ├── clustering.py           ← DC-relative KMeans + gap statistic
    └── requirements.txt
```


## Data split across 15 clients

| Client | Day | Hours | Notes |
|--------|-----|-------|-------|
| 01 | 20250204 | 00–24h | Full day |
| 02 | 20250205 | 00–24h | Full day |
| 03 | 20250206 | 00–24h | Full day |
| 04 | 20250207 | 00–24h | Full day |
| 05 | 20250208 | 00–24h | Full day |
| 06 | 20250209 | 00–24h | Full day |
| 07 | 20250210 | 00–24h | Full day |
| 08 | 20250211 | 00–16.5h | Ends at 16h30 |
| 09 | 20250206 | 00–3.4h | Night baseline |
| 10 | 20250206 | 3.4–6.8h | Pre-dawn, few events |
| 11 | 20250206 | 6.8–10.2h | Morning rush |
| 12 | 20250206 | 10.2–13.6h | Mid-day traffic |
| 13 | 20250206 | 13.6–17.0h | Afternoon peak |
| 14 | 20250206 | 17.0–20.4h | Evening, mixed |
| 15 | 20250206 | 20.4–24.0h | Night wind-down |

Clients 09–15 simulate spatially displaced sensors along the pipeline
all seeing the same physical day from different angular/positional perspectives.


