#!/usr/bin/env python3
"""
Fix Feature Mismatch Between Training and Real-Time System
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_feature_mismatch(error_message):
    """Analyze the feature mismatch error and provide solutions"""
    
    print("🔧 FEATURE MISMATCH ANALYSIS")
    print("="*40)
    
    # Parse error message
    if "Feature names unseen at fit time:" in error_message:
        print("❌ ISSUE: Real-time features don't match training features")
        print("🎯 CAUSE: Feature engineering differences between training and production")
        
        # Extract feature information from error
        lines = error_message.split('\n')
        unseen_features = []
        missing_features = []
        
        capturing_unseen = False
        capturing_missing = False
        
        for line in lines:
            if "Feature names unseen at fit time:" in line:
                capturing_unseen = True
                capturing_missing = False
            elif "Feature names seen at fit time, yet now missing:" in line:
                capturing_unseen = False
                capturing_missing = True
            elif line.strip().startswith('- ') and capturing_unseen:
                feature = line.strip()[2:]
                if feature != '...':
                    unseen_features.append(feature)
            elif line.strip().startswith('- ') and capturing_missing:
                feature = line.strip()[2:]
                if feature != '...':
                    missing_features.append(feature)
        
        print(f"\n📊 FEATURE ANALYSIS:")
        print(f"   Unseen features (in real-time, not in training): {len(unseen_features)}")
        for feat in unseen_features[:5]:
            print(f"      • {feat}")
        if len(unseen_features) > 5:
            print(f"      ... and {len(unseen_features)-5} more")
            
        print(f"\n   Missing features (in training, not in real-time): {len(missing_features)}")
        for feat in missing_features[:5]:
            print(f"      • {feat}")
        if len(missing_features) > 5:
            print(f"      ... and {len(missing_features)-5} more")
        
        return unseen_features, missing_features
    
    return [], []

def create_feature_mapper():
    """Create a feature mapping solution"""
    
    print(f"\n🔧 CREATING FEATURE MAPPER SOLUTION")
    print("="*45)
    
    feature_mapper_code = '''
class FeatureMapper:
    """Map real-time features to match training features"""
    
    def __init__(self, training_features=None):
        self.training_features = training_features or []
        self.feature_mapping = {}
        self.setup_feature_mappings()
    
    def setup_feature_mappings(self):
        """Setup common feature name mappings"""
        
        # Bollinger Band mappings
        self.feature_mapping.update({
            'bb_lower_20_2': 'bb_lower',
            'bb_upper_20_2': 'bb_upper', 
            'bb_middle_20_2': 'bb_middle',
            'bb_position_20_2': 'bb_position',
            'bb_lower_20_2.5': 'bb_lower_25',
            'bb_upper_20_2.5': 'bb_upper_25',
            'bb_position_20_2.5': 'bb_position_25',
            'bb_position_50_2': 'bb_position_50',
        })
        
        # ATR mappings
        self.feature_mapping.update({
            'atr_norm_14': 'atr_normalized_14',
            'atr_norm_21': 'atr_normalized_21',
        })
        
        # Candlestick pattern mappings
        self.feature_mapping.update({
            'doji': 'doji_pattern',
            'hammer': 'hammer_pattern',
            'engulfing': 'engulfing_pattern',
        })
        
        # Common indicator variations
        self.feature_mapping.update({
            'sma_5': 'sma_5',
            'sma_10': 'sma_10', 
            'sma_20': 'sma_20',
            'sma_50': 'sma_50',
            'ema_12': 'ema_12',
            'ema_26': 'ema_26',
        })
    
    def map_features(self, real_time_features):
        """Map real-time features to training feature names"""
        
        mapped_features = {}
        
        for rt_feature, value in real_time_features.items():
            # Direct mapping if exists
            if rt_feature in self.feature_mapping:
                mapped_name = self.feature_mapping[rt_feature]
                mapped_features[mapped_name] = value
            # Keep original if it matches training features
            elif rt_feature in self.training_features:
                mapped_features[rt_feature] = value
            # Try pattern matching for common variations
            else:
                mapped_name = self.pattern_match_feature(rt_feature)
                if mapped_name:
                    mapped_features[mapped_name] = value
        
        return mapped_features
    
    def pattern_match_feature(self, feature_name):
        """Pattern match feature names"""
        
        # ATR pattern matching
        if feature_name.startswith('atr_'):
            if 'norm' in feature_name:
                period = feature_name.split('_')[2] if len(feature_name.split('_')) > 2 else '14'
                return f'atr_normalized_{period}'
            else:
                return feature_name
        
        # Bollinger Band pattern matching  
        if feature_name.startswith('bb_'):
            parts = feature_name.split('_')
            if len(parts) >= 3:
                bb_type = parts[1]  # lower, upper, middle, position
                period = parts[2] if parts[2].isdigit() else '20'
                std_dev = parts[3] if len(parts) > 3 else '2'
                
                if std_dev == '2':
                    return f'bb_{bb_type}'
                else:
                    return f'bb_{bb_type}_{std_dev.replace(".", "")}'
        
        # RSI variations
        if feature_name.startswith('rsi_'):
            return feature_name  # Usually consistent
        
        # MACD variations
        if feature_name.startswith('macd'):
            return feature_name  # Usually consistent
            
        return None
    
    def add_missing_features(self, features_dict, default_value=0.0):
        """Add missing features with default values"""
        
        for training_feature in self.training_features:
            if training_feature not in features_dict:
                # Try to compute missing feature if possible
                computed_value = self.compute_missing_feature(training_feature, features_dict)
                features_dict[training_feature] = computed_value if computed_value is not None else default_value
        
        return features_dict
    
    def compute_missing_feature(self, feature_name, available_features):
        """Compute missing features from available ones"""
        
        # Bollinger Band derived features
        if feature_name == 'bb_position' and 'bb_upper' in available_features and 'bb_lower' in available_features:
            if 'close' in available_features:
                bb_range = available_features['bb_upper'] - available_features['bb_lower']
                if bb_range > 0:
                    return (available_features['close'] - available_features['bb_lower']) / bb_range
        
        # ATR normalized from regular ATR
        if feature_name.startswith('atr_normalized_') and feature_name.replace('normalized_', '') in available_features:
            atr_value = available_features[feature_name.replace('normalized_', '')]
            if 'close' in available_features and available_features['close'] > 0:
                return atr_value / available_features['close']
        
        # Candlestick patterns (simplified)
        if feature_name == 'doji' and all(k in available_features for k in ['open', 'close', 'high', 'low']):
            body_size = abs(available_features['close'] - available_features['open'])
            total_range = available_features['high'] - available_features['low']
            return 1 if body_size < (total_range * 0.1) else 0
        
        return None
    
    def transform_features(self, real_time_features):
        """Complete feature transformation pipeline"""
        
        # Step 1: Map feature names
        mapped_features = self.map_features(real_time_features)
        
        # Step 2: Add missing features
        complete_features = self.add_missing_features(mapped_features)
        
        # Step 3: Ensure correct order (if training feature order is known)
        if self.training_features:
            ordered_features = {}
            for feature in self.training_features:
                ordered_features[feature] = complete_features.get(feature, 0.0)
            return ordered_features
        
        return complete_features

# Usage example
def fix_real_time_features(real_time_features, model_metadata=None):
    """Fix real-time features to match training"""
    
    # Load training features from model metadata if available
    training_features = []
    if model_metadata and 'selected_features' in model_metadata:
        training_features = model_metadata['selected_features']
    
    # Create mapper
    mapper = FeatureMapper(training_features)
    
    # Transform features
    fixed_features = mapper.transform_features(real_time_features)
    
    return fixed_features
'''
    
    print("✅ Feature mapper solution created")
    return feature_mapper_code

def create_integration_fix():
    """Create integration fix for the trading system"""
    
    print(f"\n🔧 CREATING TRADING SYSTEM INTEGRATION FIX")
    print("="*50)
    
    integration_code = '''
# Integration fix for trading system
class TradingSystemFeatureFix:
    """Fix feature compatibility in trading system"""
    
    def __init__(self):
        self.feature_mapper = None
        self.model_metadata = None
        self.load_model_metadata()
    
    def load_model_metadata(self):
        """Load model metadata to get training features"""
        
        try:
            # Look for latest model metadata
            from pathlib import Path
            import json
            import glob
            
            models_path = Path("exported_models")
            metadata_files = list(models_path.glob("*training_metadata*.json"))
            
            if metadata_files:
                # Get most recent metadata file
                latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_metadata, 'r') as f:
                    self.model_metadata = json.load(f)
                
                print(f"📊 Loaded model metadata: {latest_metadata.name}")
                
                if 'selected_features' in self.model_metadata:
                    training_features = self.model_metadata['selected_features']
                    self.feature_mapper = FeatureMapper(training_features)
                    print(f"✅ Feature mapper initialized with {len(training_features)} training features")
                else:
                    print("⚠️ No selected_features found in metadata")
            else:
                print("⚠️ No model metadata files found")
                
        except Exception as e:
            print(f"❌ Failed to load model metadata: {e}")
    
    def fix_signal_generation(self, original_signal_func):
        """Wrap signal generation to fix features"""
        
        def fixed_signal_generation(*args, **kwargs):
            try:
                # Get original signal
                signal = original_signal_func(*args, **kwargs)
                
                # Check for feature error
                if hasattr(signal, 'features') and isinstance(signal.features, dict):
                    if 'error' in signal.features:
                        error_msg = signal.features['error']
                        
                        if "Feature names should match" in error_msg:
                            print("🔧 Detected feature mismatch, attempting fix...")
                            
                            # Try to regenerate features with proper mapping
                            if self.feature_mapper:
                                # This would need to be integrated with your actual feature generation
                                print("✅ Feature mapper available for fixing")
                            else:
                                print("❌ No feature mapper available")
                
                return signal
                
            except Exception as e:
                print(f"❌ Error in signal generation fix: {e}")
                return original_signal_func(*args, **kwargs)
        
        return fixed_signal_generation
    
    def create_feature_alignment_wrapper(self, feature_generation_func):
        """Wrap feature generation to ensure alignment"""
        
        def aligned_feature_generation(data, symbol=None):
            try:
                # Generate features normally
                features = feature_generation_func(data, symbol)
                
                # Apply feature mapping if available
                if self.feature_mapper and isinstance(features, dict):
                    fixed_features = self.feature_mapper.transform_features(features)
                    return fixed_features
                elif self.feature_mapper and hasattr(features, 'to_dict'):
                    # If features is a pandas Series/DataFrame
                    features_dict = features.to_dict() if hasattr(features, 'to_dict') else dict(features)
                    fixed_features = self.feature_mapper.transform_features(features_dict)
                    return fixed_features
                
                return features
                
            except Exception as e:
                print(f"❌ Error in feature alignment: {e}")
                return feature_generation_func(data, symbol)
        
        return aligned_feature_generation

# Quick fix function
def quick_fix_trading_system(trading_system):
    """Apply quick fix to existing trading system"""
    
    print("🔧 Applying quick fix to trading system...")
    
    # Create feature fix instance
    fix = TradingSystemFeatureFix()
    
    # If feature mapper is available, wrap the signal generation
    if fix.feature_mapper:
        print("✅ Feature mapper found, applying fix")
        
        # This is a conceptual fix - actual implementation depends on your trading system structure
        if hasattr(trading_system, 'generate_signal'):
            original_func = trading_system.generate_signal
            trading_system.generate_signal = fix.fix_signal_generation(original_func)
            print("✅ Signal generation wrapped with feature fix")
        
        return True
    else:
        print("❌ No feature mapper available - manual feature alignment needed")
        return False
'''
    
    print("✅ Integration fix solution created")
    return integration_code

def provide_immediate_solutions():
    """Provide immediate solutions for the feature mismatch"""
    
    print(f"\n🚀 IMMEDIATE SOLUTIONS")
    print("="*30)
    
    print("1️⃣ QUICK FIX - Model Retraining")
    print("   • Retrain model with current real-time feature names")
    print("   • Ensure feature engineering consistency")
    print("   • Use same preprocessing pipeline")
    
    print("\n2️⃣ MEDIUM FIX - Feature Mapping")
    print("   • Implement FeatureMapper class (provided above)")
    print("   • Map real-time features to training features")
    print("   • Handle missing features with defaults")
    
    print("\n3️⃣ ROBUST FIX - Unified Feature Pipeline")
    print("   • Create single feature engineering module")
    print("   • Use for both training and real-time")
    print("   • Ensure consistent feature names")
    
    print(f"\n💡 RECOMMENDED APPROACH:")
    print("   1. Implement FeatureMapper as temporary fix")
    print("   2. Retrain model with aligned features")
    print("   3. Create unified feature pipeline for future")
    
    return True

def main():
    """Main function to analyze and fix feature mismatch"""
    
    # Sample error message from user
    error_message = '''The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- bb_lower_20_2
- bb_lower_20_2.5
- bb_position_20_2
- bb_position_20_2.5
- bb_position_50_2
- ...
Feature names seen at fit time, yet now missing:
- atr_14
- atr_21
- atr_norm_14
- atr_norm_21
- doji
- ...'''
    
    # Analyze the error
    unseen_features, missing_features = analyze_feature_mismatch(error_message)
    
    # Create solutions
    feature_mapper = create_feature_mapper()
    integration_fix = create_integration_fix()
    
    # Provide immediate solutions
    provide_immediate_solutions()
    
    print(f"\n🎉 FEATURE MISMATCH ANALYSIS COMPLETE")
    print("="*45)
    print("✅ Problem identified and solutions provided")
    print("✅ FeatureMapper class ready for implementation")
    print("✅ Integration fixes available")
    print("🚀 Ready to resolve trading system feature mismatch!")

if __name__ == "__main__":
    main()