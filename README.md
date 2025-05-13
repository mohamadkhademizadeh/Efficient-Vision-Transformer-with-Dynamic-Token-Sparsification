# Efficient-Vision-Transformer-with-Dynamic-Token-Sparsification
efficient_vit/
├── configs/               # Configuration files
│   └── default.yaml       # Hyperparameters
├── data/                  # Dataset loading
│   └── cifar.py
├── models/                # Model definitions
│   ├── vit.py             # Vision Transformer
│   └── scorer.py          # Token importance predictor
├── scripts/
│   ├── train.py           # Training script
│   └── profile.py         # Latency/FLOPs measurement
├── utils/                 # Helpers
│   ├── logger.py          # Metrics tracking
│   └── pruning.py         # Token dropping logic
└── README.md              # Research-grade documentation
