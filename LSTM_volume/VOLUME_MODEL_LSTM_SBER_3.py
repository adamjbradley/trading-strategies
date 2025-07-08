import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MetaTrader5 as mt5
import logging
import os
from datetime import datetime
import traceback
import json

terminal_path = "C:/Program Files/MetaTrader 5/terminal64.exe"

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lookback):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.lookback = lookback
        
    def __len__(self):
        return len(self.X) - self.lookback
        
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.lookback],
            self.y[idx + self.lookback]
        )

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        output = self.linear(dropped)
        return output

class VolumeAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger('VolumeAnalyzer')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_logging(self):
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_filename = os.path.join(log_dir, f'volume_analyzer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger('VolumeAnalyzer')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def setup_environment(self):
        try:
            self.logger.info("Initializing MetaTrader5...")
            if not mt5.initialize(path=terminal_path):
                self.logger.error(f"MT5 initialization error: {mt5.last_error()}")
                return False
            self.logger.info("MetaTrader5 successfully initialized")
            return True
        except Exception as e:
            self.logger.error(f"Critical error during MT5 initialization: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    def get_mt5_data(self, symbol, timeframe, start_date, end_date):
        try:
            self.logger.info(f"Requesting MT5 data: {symbol}, {timeframe}, {start_date} - {end_date}")
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get data: {mt5.last_error()}")
                return None
                
            self.logger.info(f"Retrieved {len(rates)} records")
            df = pd.DataFrame(rates)
            
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")
            
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='s')
            
            self.logger.debug(f"Data structure:\n{df.head().to_string()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error while getting data: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def preprocess_data(self, df):
        try:
            self.logger.info("Starting data preprocessing")
            self.logger.debug(f"Initial data shape: {df.shape}")
            
            # Basic volume indicators
            df['vol_ma5'] = df['real_volume'].rolling(window=5).mean()
            df['vol_ma20'] = df['real_volume'].rolling(window=20).mean()
            df['vol_ratio'] = df['real_volume'] / df['vol_ma20']
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Additional indicators for prediction
            df['price_momentum'] = df['close'].pct_change(24)
            df['volume_momentum'] = df['real_volume'].pct_change(24)
            df['volume_volatility'] = df['real_volume'].pct_change().rolling(24).std()
            df['price_volume_correlation'] = df['price_change'].rolling(24).corr(df['real_volume'].pct_change())
            
            # Target variable - price change after 24 hours
            df['target_return'] = df['close'].shift(-24) / df['close'] - 1
            
            initial_rows = len(df)
            df.dropna(inplace=True)
            dropped_rows = initial_rows - len(df)
            
            self.logger.info(f"Rows with missing values removed: {dropped_rows}")
            self.logger.debug(f"Final data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def detect_volume_anomalies(self, df):
        try:
            self.logger.info("Starting volume anomaly detection")
            
            scaler = StandardScaler()
            volume_normalized = scaler.fit_transform(df[['real_volume']])
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df['is_anomaly'] = iso_forest.fit_predict(volume_normalized)
            
            anomaly_count = len(df[df['is_anomaly'] == -1])
            self.logger.info(f"Anomalies found: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def cluster_volumes(self, df, n_clusters=10):
        try:
            self.logger.info(f"Starting volume clustering (k={n_clusters})")
            
            features = ['real_volume', 'vol_ratio', 'volatility']
            X = StandardScaler().fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            df['volume_cluster'] = kmeans.fit_predict(X)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error during clustering: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def prepare_model_data(self, df, test_size=0.2, lookback=24):
        try:
            features = [
                'vol_ratio', 'vol_ma5', 'volatility', 'volume_cluster', 'is_anomaly',
                'price_momentum', 'volume_momentum', 'volume_volatility', 'price_volume_correlation'
            ]
            
            X = df[features].values
            y = df['target_return'].values
            
            # Feature normalization
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train-test split
            train_size = int(len(X_scaled) * (1 - test_size))
            
            X_train = X_scaled[:train_size]
            X_test = X_scaled[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            # Create PyTorch datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, lookback)
            test_dataset = TimeSeriesDataset(X_test, y_test, lookback)
            
            return train_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Error during model data preparation: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None, None

    def train_model(self, train_dataset, test_dataset, batch_size=32, epochs=10):
        try:
            self.logger.info("Starting model training")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            input_size = train_dataset.X.shape[1]
            self.model = LSTMModel(input_size=input_size).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            
            best_val_loss = float('inf')
            early_stopping_rounds = 10
            counter = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_losses = []
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch.unsqueeze(1))
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Validation
                self.model.eval()
                val_losses = []
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        predictions = self.model(X_batch)
                        val_loss = criterion(predictions, y_batch.unsqueeze(1))
                        
                        val_losses.append(val_loss.item())
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(y_batch.cpu().numpy())
                
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    counter += 1
                    if counter >= early_stopping_rounds:
                        self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            # Load best model
            self.model.load_state_dict(torch.load('best_model.pth'))
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def backtest_prediction_strategy(self, df, lookback=24):
        try:
            self.logger.info("Starting strategy backtesting")
            
            # Data preparation
            train_dataset, test_dataset = self.prepare_model_data(df, lookback=lookback)
            if train_dataset is None:
                return None
            
            # Model training
            model = self.train_model(train_dataset, test_dataset)
            if model is None:
                return None
            
            # Trading signals based on predictions
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(lookback, len(df)):
                    features = df.iloc[i-lookback:i][['vol_ratio', 'vol_ma5', 'volatility', 'volume_cluster', 'is_anomaly',
                                                      'price_momentum', 'volume_momentum', 'volume_volatility', 'price_volume_correlation']].values
                    features_scaled = self.feature_scaler.transform(features)
                    features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
                    pred = model(features_tensor)
                    predictions.append(pred.cpu().numpy()[0][0])
            
            # Add predictions to dataframe
            df['predicted_return'] = [float('nan')] * lookback + predictions
            
            # Signal generation
            df['signal'] = 0
            signal_threshold = 0.001  # Signal threshold of 0.1%
            df.loc[df['predicted_return'] > signal_threshold, 'signal'] = 1
            df.loc[df['predicted_return'] < -signal_threshold, 'signal'] = -1
            
            # Calculate returns
            df['strategy_returns'] = df['signal'].shift(24) * df['price_change']
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            
            # Trading statistics
            total_trades = len(df[df['signal'] != 0])
            winning_trades = len(df[df['strategy_returns'] > 0])
            losing_trades = len(df[df['strategy_returns'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            final_return = df['cumulative_returns'].iloc[-1] - 1
            max_drawdown = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
            sharpe_ratio = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std()
            
            stats = {
                'Total trades': total_trades,
                'Winning trades': winning_trades,
                'Losing trades': losing_trades,
                'Win rate': f'{win_rate:.2%}',
                'Final return': f'{final_return:.2%}',
                'Maximum drawdown': f'{max_drawdown:.2%}',
                'Sharpe ratio': f'{sharpe_ratio:.2f}'
            }
            
            self.logger.info("Trading statistics:")
            self.logger.info(json.dumps(stats, indent=2))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def visualize_results(self, df):
        try:
            self.logger.info("Creating results visualization")
            
            # Calculate the aspect ratio to maintain proportions
            aspect_ratio = 15 / 12  # Original aspect ratio from figsize=(15, 12)
            new_width = 750
            new_height = int(new_width / aspect_ratio)
            
            plt.figure(figsize=(new_width / 100, new_height / 100))  # Convert to inches
            
            # Price and signals plot
            plt.subplot(3, 1, 1)
            plt.plot(df['time'], df['close'], 'k-', label='Price', alpha=0.7)
            plt.scatter(df[df['signal'] == 1]['time'], 
                        df[df['signal'] == 1]['close'],
                        marker='^', color='g', label='Buy', alpha=0.7)
            plt.scatter(df[df['signal'] == -1]['time'],
                        df[df['signal'] == -1]['close'],
                        marker='v', color='r', label='Sell', alpha=0.7)
            plt.title('Price and Trading Signals')
            plt.legend()
            plt.grid(True)
            
            # Predicted returns plot
            plt.subplot(3, 1, 2)
            plt.plot(df['time'], df['predicted_return'], 'b-', label='Predicted Return')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title('Predicted Returns')
            plt.legend()
            plt.grid(True)
            
            # Cumulative returns plot
            plt.subplot(3, 1, 3)
            plt.plot(df['time'], df['cumulative_returns'], 'g-', label='Cumulative Returns')
            plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
            plt.fill_between(df['time'], 
                            1,
                            df['cumulative_returns'],
                            where=(df['cumulative_returns'] < 1),
                            color='red',
                            alpha=0.3,
                            label='Drawdown')
            plt.title('Strategy Cumulative Returns')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Create plots directory if it doesn't exist
            plots_dir = 'plots'
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            plot_path = os.path.join(plots_dir, f'prediction_strategy_{datetime.now().strftime("%Y%m%d")}.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')  # Adjust dpi to maintain quality
            self.logger.info(f"Plot saved to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error during visualization: {str(e)}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

def main():
    analyzer = VolumeAnalyzer()
    
    try:
        # Initialization
        if not analyzer.setup_environment():
            return False
        
        # Analysis parameters
        symbol = "GOOG"
        timeframe = mt5.TIMEFRAME_H1
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365)
        end_date = pd.Timestamp.now()
        
        # Get and process data
        df = analyzer.get_mt5_data(symbol, timeframe, start_date, end_date)
        if df is None:
            return False
            
        df = analyzer.preprocess_data(df)
        if df is None:
            return False
            
        # Data analysis
        df = analyzer.detect_volume_anomalies(df)
        if df is None:
            return False
            
        df = analyzer.cluster_volumes(df)
        if df is None:
            return False
        
        # Backtest prediction-based strategy
        df = analyzer.backtest_prediction_strategy(df)
        if df is None:
            return False
            
        # Visualize results
        analyzer.visualize_results(df)
        
        # Close connection
        mt5.shutdown()
        analyzer.logger.info("Analysis successfully completed")
        
    except Exception as e:
        analyzer.logger.error(f"Critical error in main: {str(e)}")
        analyzer.logger.debug(f"Traceback: {traceback.format_exc()}")
        mt5.shutdown()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("Program ended with errors. Check logs for additional information.")
    else:
        print("Program completed successfully.")
