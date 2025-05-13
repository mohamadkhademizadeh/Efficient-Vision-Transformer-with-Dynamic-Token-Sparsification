import hydra
from omegaconf import DictConfig
from models.vit import ViTWithPruning
from data.cifar import load_cifar10

@hydra.main(config_path="../configs", config_name="default")
def train(cfg: DictConfig):
    # Load data
    train_loader, test_loader = load_cifar10(batch_size=cfg.batch_size)
    
    # Model
    model = ViTWithPruning(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=cfg.dim,
        depth=cfg.depth,
        prune_layers=cfg.prune_layers,
        keep_ratios=cfg.keep_ratios
    ).cuda()
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            
            # Forward
            logits = model(x)
            loss = criterion(logits, y)
            
            # Sparsity regularization
            if cfg.sparsity_weight > 0:
                # Compute actual sparsity (simplified)
                sparsity = 1.0 - (model.token_mask.sum() / (model.token_mask.shape[0] * model.token_mask.shape[1]))
                loss += cfg.sparsity_weight * (sparsity - cfg.target_sparsity)**2
            
            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Evaluation
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}: Loss={loss.item():.3f}, Acc={acc:.2f}%")

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
    return 100 * correct / len(loader.dataset)
