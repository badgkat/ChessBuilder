import sys, os, torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QTextEdit, QLineEdit, QLabel, QFormLayout, QHBoxLayout, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal

# Import your training components.
from .selfplay import generate_selfplay_data
from .dataset import ChessDataset
from .model import ChessNet

# Worker thread that runs training.
class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # Emits current progress (move count).
    status_signal = pyqtSignal(str)    # Emits status string with move counter and avg reward.

    def __init__(self, num_iterations, games_per_iter, epochs_per_iter, batch_size, parent=None):
        super().__init__(parent)
        self.num_iterations = num_iterations
        self.games_per_iter = games_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.batch_size = batch_size

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessNet(num_channels=13, policy_size=8513).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.log_signal.emit(f"Loaded model and optimizer from {checkpoint_path}")
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Total progress steps: each iteration has a data generation phase (games_per_iter steps)
        # and a training phase (epochs_per_iter steps).
        total_steps = self.num_iterations * (self.games_per_iter + self.epochs_per_iter)
        current_step = 0

        for iteration in range(self.num_iterations):
            if self.isInterruptionRequested():
                self.log_signal.emit("Training interrupted.")
                break

            self.log_signal.emit(f"\n=== Iteration {iteration}: Generating self-play data ===")
            # Generate one game at a time.
            for game in range(self.games_per_iter):
                if self.isInterruptionRequested():
                    self.log_signal.emit("Training interrupted during data generation.")
                    break
                generate_selfplay_data(num_games=1, model=model, device=device)
                current_step += 1
                self.progress_signal.emit(current_step)
            
            self.log_signal.emit(f"=== Iteration {iteration}: Training model ===")
            # Load newly generated data.
            dataset = ChessDataset()  # Loads training_data.npz.
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            model.train()
            for epoch in range(self.epochs_per_iter):
                if self.isInterruptionRequested():
                    self.log_signal.emit("Training interrupted during training phase.")
                    break
                epoch_loss = 0.0
                total_reward = 0.0
                total_samples = 0
                for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
                    states = states.to(device)
                    policy_targets = policy_targets.to(device)
                    value_targets = value_targets.to(device)
                    
                    optimizer.zero_grad()
                    policy_pred, value_pred = model(states)
                    loss_policy = torch.nn.functional.mse_loss(policy_pred, policy_targets)
                    loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
                    loss = loss_policy + loss_value
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    total_reward += value_targets.sum().item()
                    total_samples += value_targets.numel()
                    
                    if self.isInterruptionRequested():
                        break
                avg_loss = epoch_loss / len(dataloader)
                avg_reward = total_reward / total_samples if total_samples > 0 else 0
                self.log_signal.emit(f"Iteration {iteration} Epoch {epoch} Average Loss: {avg_loss:.6f}, Average Reward: {avg_reward:.6f}")
                current_step += 1
                self.progress_signal.emit(current_step)
                self.status_signal.emit(
                    f"Iteration {iteration} Epoch {epoch}: Move {current_step} of {total_steps}, Avg Reward: {avg_reward:.3f}"
                )
            
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            self.log_signal.emit(f"Iteration {iteration} checkpoint saved.")
        
        self.finished_signal.emit("Iterative training complete.")

# Main UI.
class TrainingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess AI Training")
        self.worker = None
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        self.iterations_input = QLineEdit("10")
        self.games_input = QLineEdit("50")
        self.epochs_input = QLineEdit("5")
        self.batch_input = QLineEdit("32")
        
        form_layout.addRow("Iterations:", self.iterations_input)
        form_layout.addRow("Games/Iteration:", self.games_input)
        form_layout.addRow("Epochs/Iteration:", self.epochs_input)
        form_layout.addRow("Batch Size:", self.batch_input)
        
        layout.addLayout(form_layout)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # Progress bar setup.
        self.progress_bar = QProgressBar()
        # Its range will be set based on total steps.
        self.progress_bar.setRange(0, 1)
        layout.addWidget(self.progress_bar)
        
        # Status label for move counter and reward info.
        self.status_label = QLabel("Status: ")
        layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        central_widget.setLayout(layout)
        
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        
    def start_training(self):
        num_iterations = int(self.iterations_input.text())
        games_per_iter = int(self.games_input.text())
        epochs_per_iter = int(self.epochs_input.text())
        batch_size = int(self.batch_input.text())
        
        total_steps = num_iterations * (games_per_iter + epochs_per_iter)
        self.progress_bar.setRange(0, total_steps)
        
        self.worker = TrainingWorker(num_iterations, games_per_iter, epochs_per_iter, batch_size)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.append_log("Training started.")

    def stop_training(self):
        if self.worker:
            self.worker.requestInterruption()
            self.append_log("Stop requested. The training will stop after the current operation.")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setValue(0)
    
    def append_log(self, message):
        self.log_text.append(message)
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, status):
        self.status_label.setText(f"Status: {status}")
        
    def training_finished(self, message):
        self.append_log(message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("Status: Training complete.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingUI()
    window.show()
    sys.exit(app.exec_())
