"""
Lead 期货数据准备模块

功能：
1. 数据清洗和预处理
2. 特征工程（技术指标、滞后特征等）
3. 目标变量创建
4. 数据保存和加载
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    加载原始 CSV 数据
    
    参数:
        data_path: 原始数据文件路径
    
    返回:
        原始 DataFrame
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"成功加载数据: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗原始数据
    
    处理内容：
    1. 日期格式转换
    2. 移除价格中的逗号并转换为数值
    3. 处理涨跌幅百分比
    4. 处理成交量（K/M 后缀）
    5. 数据排序和验证
    
    参数:
        df: 原始 DataFrame
    
    返回:
        清洗后的 DataFrame
    """
    df_clean = df.copy()
    
    # 1. 处理日期：转换为 datetime 格式
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%m/%d/%Y')
    
    # 2. 处理价格列：移除逗号并转换为浮点数
    price_columns = ['Price', 'Open', 'High', 'Low']
    for col in price_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.replace('"', '')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 3. 处理涨跌幅百分比：移除 % 符号并转换为数值（小数形式）
    if 'Change %' in df_clean.columns:
        df_clean['Change %'] = df_clean['Change %'].astype(str).str.replace('%', '').str.replace('"', '')
        df_clean['Change %'] = pd.to_numeric(df_clean['Change %'], errors='coerce') / 100
    
    # 4. 处理成交量：处理 K/M 后缀
    if 'Vol.' in df_clean.columns:
        def parse_volume(vol_str):
            """解析成交量字符串，处理 K/M 后缀"""
            if pd.isna(vol_str) or vol_str == '' or vol_str == '""':
                return np.nan
            vol_str = str(vol_str).replace('"', '').strip()
            if vol_str == '':
                return np.nan
            try:
                if 'K' in vol_str.upper():
                    return float(vol_str.upper().replace('K', '')) * 1000
                elif 'M' in vol_str.upper():
                    return float(vol_str.upper().replace('M', '')) * 1000000
                else:
                    return float(vol_str)
            except:
                return np.nan
        
        df_clean['Vol.'] = df_clean['Vol.'].apply(parse_volume)
        df_clean = df_clean.rename(columns={'Vol.': 'Volume'})
    
    # 5. 按日期排序（从早到晚）
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    # 6. 设置日期为索引
    df_clean = df_clean.set_index('Date')
    
    # 7. 重命名列（使用更标准的命名）
    column_mapping = {
        'Price': 'Close',
        'Change %': 'Change_Pct'
    }
    df_clean = df_clean.rename(columns=column_mapping)
    
    # 8. 验证和修复价格逻辑异常
    issues = []
    
    # 检查并修复 High < Low 的情况
    invalid_hl = df_clean['High'] < df_clean['Low']
    if invalid_hl.any():
        n_invalid = invalid_hl.sum()
        issues.append(f"High < Low: {n_invalid} 条记录")
        # 交换 High 和 Low
        df_clean.loc[invalid_hl, ['High', 'Low']] = df_clean.loc[invalid_hl, ['Low', 'High']].values
    
    # 检查并修复 High < Close 的情况
    invalid_hc = df_clean['High'] < df_clean['Close']
    if invalid_hc.any():
        n_invalid = invalid_hc.sum()
        issues.append(f"High < Close: {n_invalid} 条记录")
        # 将 High 设为 Close
        df_clean.loc[invalid_hc, 'High'] = df_clean.loc[invalid_hc, 'Close']
    
    # 检查并修复 High < Open 的情况
    invalid_ho = df_clean['High'] < df_clean['Open']
    if invalid_ho.any():
        n_invalid = invalid_ho.sum()
        issues.append(f"High < Open: {n_invalid} 条记录")
        # 将 High 设为 Open
        df_clean.loc[invalid_ho, 'High'] = df_clean.loc[invalid_ho, 'Open']
    
    # 检查并修复 Low > Close 的情况
    invalid_lc = df_clean['Low'] > df_clean['Close']
    if invalid_lc.any():
        n_invalid = invalid_lc.sum()
        issues.append(f"Low > Close: {n_invalid} 条记录")
        # 将 Low 设为 Close
        df_clean.loc[invalid_lc, 'Low'] = df_clean.loc[invalid_lc, 'Close']
    
    # 检查并修复 Low > Open 的情况
    invalid_lo = df_clean['Low'] > df_clean['Open']
    if invalid_lo.any():
        n_invalid = invalid_lo.sum()
        issues.append(f"Low > Open: {n_invalid} 条记录")
        # 将 Low 设为 Open
        df_clean.loc[invalid_lo, 'Low'] = df_clean.loc[invalid_lo, 'Open']
    
    if issues:
        print(f"  发现并修复了以下数据异常: {', '.join(issues)}")
    else:
        print("  数据逻辑验证通过")
    
    print(f"数据清洗完成: {df_clean.shape}")
    return df_clean


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建技术指标特征（简化版 - 仅包含指定特征）
    
    特征列表:
    1. 1-day return
    2. 7-day return
    3. 14-day return
    4. 7-day volatility
    5. 14-day volatility
    6. 21-day volatility
    7. Volume (log-scaled)
    8. Volume change %
    9. Price / MA7
    10. Price / MA30
    11. MA14 / MA30
    12. RSI
    13. MACD oscillator
    14. BB band (position)
    
    参数:
        df: 清洗后的 DataFrame
    
    返回:
        包含特征的 DataFrame
    """
    df_features = df.copy()
    
    # 1. Returns (1-day, 7-day, 14-day)
    df_features['Returns_1d'] = df_features['Close'].pct_change(1)  # 1-day return
    df_features['Returns_7d'] = df_features['Close'].pct_change(7)  # 7-day return
    df_features['Returns_14d'] = df_features['Close'].pct_change(14)  # 14-day return
    
    # 2. Volatility (7-day, 14-day, 21-day)
    # Use Returns_1d to calculate rolling volatility
    df_features['Volatility_7d'] = df_features['Returns_1d'].rolling(window=7).std() * np.sqrt(252)
    df_features['Volatility_14d'] = df_features['Returns_1d'].rolling(window=14).std() * np.sqrt(252)
    df_features['Volatility_21d'] = df_features['Returns_1d'].rolling(window=21).std() * np.sqrt(252)
    
    # 3. Moving Averages (for ratios)
    df_features['MA_7'] = df_features['Close'].rolling(window=7).mean()
    df_features['MA_14'] = df_features['Close'].rolling(window=14).mean()
    df_features['MA_30'] = df_features['Close'].rolling(window=30).mean()
    
    # 4. Price / MA ratios
    df_features['Price_MA7_Ratio'] = df_features['Close'] / df_features['MA_7']
    df_features['Price_MA30_Ratio'] = df_features['Close'] / df_features['MA_30']
    
    # 5. MA14 / MA30 ratio
    df_features['MA14_MA30_Ratio'] = df_features['MA_14'] / df_features['MA_30']
    
    # 6. RSI (14-day)
    delta = df_features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # 7. MACD oscillator (EMA12 - EMA26)
    df_features['EMA_12'] = df_features['Close'].ewm(span=12, adjust=False).mean()
    df_features['EMA_26'] = df_features['Close'].ewm(span=26, adjust=False).mean()
    df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
    
    # 8. Bollinger Bands (BB Position - position within the band)
    df_features['BB_Middle'] = df_features['Close'].rolling(window=20).mean()
    bb_std = df_features['Close'].rolling(window=20).std()
    df_features['BB_Upper'] = df_features['BB_Middle'] + 2 * bb_std
    df_features['BB_Lower'] = df_features['BB_Middle'] - 2 * bb_std
    df_features['BB_Position'] = (df_features['Close'] - df_features['BB_Lower']) / (df_features['BB_Upper'] - df_features['BB_Lower'])
    
    # 9. Volume features (if Volume column exists)
    if 'Volume' in df_features.columns:
        # Volume (log-scaled) - add small value to avoid log(0)
        df_features['Volume_Log'] = np.log(df_features['Volume'] + 1)
        # Volume change %
        df_features['Volume_Change_Pct'] = df_features['Volume'].pct_change()
    else:
        df_features['Volume_Log'] = np.nan
        df_features['Volume_Change_Pct'] = np.nan
    
    # Keep only the required features plus original price/volume columns for reference
    # We'll keep the original columns (Close, Open, High, Low, Volume, Change_Pct) 
    # and the new features, but remove intermediate calculation columns
    columns_to_keep = [
        'Close', 'Open', 'High', 'Low', 'Volume', 'Change_Pct',  # Original columns
        'Returns_1d', 'Returns_7d', 'Returns_14d',  # Returns
        'Volatility_7d', 'Volatility_14d', 'Volatility_21d',  # Volatility
        'Volume_Log', 'Volume_Change_Pct',  # Volume features
        'Price_MA7_Ratio', 'Price_MA30_Ratio', 'MA14_MA30_Ratio',  # MA ratios
        'RSI', 'MACD', 'BB_Position'  # Technical indicators
    ]
    
    # Select only the columns that exist
    available_columns = [col for col in columns_to_keep if col in df_features.columns]
    df_features = df_features[available_columns]
    
    print(f"特征创建完成: {df_features.shape}")
    print(f"特征数量: {len(df_features.columns)}")
    print(f"特征列表: {[col for col in df_features.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Change_Pct']]}")
    
    return df_features


def create_targets(df: pd.DataFrame, 
                  forward_days: int = 30) -> pd.DataFrame:
    """
    创建目标变量：30 天后的涨跌分类
    
    参数:
        df: 包含特征的 DataFrame
        forward_days: 预测未来多少天的涨跌（默认 30 天）
    
    返回:
        包含目标变量的 DataFrame
    """
    df_targets = df.copy()
    
    # 计算 30 天后的收益率
    df_targets['Forward_Return_30'] = df_targets['Close'].shift(-forward_days) / df_targets['Close'] - 1
    
    # 创建二分类目标：涨=1，跌=0
    # 如果未来 30 天收益率 > 0，则为涨（1），否则为跌（0）
    df_targets['Target_30d_Up'] = (df_targets['Forward_Return_30'] > 0).astype(int)
    
    # 为了便于理解，也可以创建一个更明确的列名
    df_targets['Target'] = df_targets['Target_30d_Up']
    
    # 统计目标变量分布
    target_counts = df_targets['Target'].value_counts()
    print(f"目标变量创建完成（预测 {forward_days} 天后涨跌）")
    print(f"  目标变量: Target (30天后涨跌分类)")
    print(f"  涨 (1): {target_counts.get(1, 0)} 条 ({target_counts.get(1, 0)/len(df_targets)*100:.1f}%)")
    print(f"  跌 (0): {target_counts.get(0, 0)} 条 ({target_counts.get(0, 0)/len(df_targets)*100:.1f}%)")
    print(f"  缺失值: {df_targets['Target'].isna().sum()} 条")
    
    return df_targets




def remove_nan_rows(df: pd.DataFrame, 
                    target_column: str = 'Target') -> pd.DataFrame:
    """
    移除包含 NaN 的行
    
    参数:
        df: DataFrame
        target_column: 目标变量列名（默认 'Target'）
    
    返回:
        清理后的 DataFrame
    """
    # 只移除目标变量为 NaN 的行，保留特征中的 NaN（后续可以填充）
    df_clean = df.dropna(subset=[target_column])
    
    removed = len(df) - len(df_clean)
    print(f"移除了 {removed} 行目标变量为 NaN 的数据 ({removed/len(df):.1%})")
    
    # 填充特征中的 NaN（使用前向填充，如果还有 NaN 则用 0）
    feature_cols = [col for col in df_clean.columns if col not in ['Target', 'Target_30d_Up', 'Forward_Return_30']]
    df_clean[feature_cols] = df_clean[feature_cols].ffill().fillna(0)
    
    remaining_nan = df_clean[feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        print(f"  填充了特征中的 NaN，剩余 {remaining_nan} 个 NaN")
    
    return df_clean


def prepare_dataset(data_path: str,
                   output_dir: str = "data",
                   forward_days: int = 30,
                   save_intermediate: bool = True) -> pd.DataFrame:
    """
    完整的数据准备流程 - 30 天后涨跌分类任务
    
    参数:
        data_path: 原始数据文件路径
        output_dir: 输出目录
        forward_days: 预测未来多少天的涨跌（默认 30 天）
        save_intermediate: 是否保存中间结果
    
    返回:
        完整的数据集 DataFrame（包含特征和目标变量）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始数据准备流程")
    print("=" * 60)
    
    # 1. 加载原始数据
    print("\n[1/5] 加载原始数据...")
    df_raw = load_raw_data(data_path)
    
    # 2. 数据清洗
    print("\n[2/5] 数据清洗...")
    df_clean = clean_data(df_raw)
    if save_intermediate:
        df_clean.to_csv(output_dir / "lead_futures_cleaned.csv")
        print(f"  已保存清洗后的数据: {output_dir / 'lead_futures_cleaned.csv'}")
    
    # 3. 创建特征
    print("\n[3/5] 创建特征...")
    df_features = create_features(df_clean)
    
    # 4. 创建目标变量（30 天后涨跌分类）
    print("\n[4/5] 创建目标变量（30 天后涨跌分类）...")
    df_with_targets = create_targets(df_features, forward_days=forward_days)
    
    # 保存带特征和目标的数据
    if save_intermediate:
        # 重置索引以便保存 Date 列
        df_with_targets_save = df_with_targets.reset_index()
        df_with_targets_save.to_csv(output_dir / "lead_futures_with_features.csv", index=False)
        print(f"  已保存带特征和目标的数据: {output_dir / 'lead_futures_with_features.csv'}")
    
    # 5. 移除 NaN 行并填充特征缺失值
    print("\n[5/5] 移除 NaN 行并填充特征缺失值...")
    df_final = remove_nan_rows(df_with_targets, target_column='Target')
    
    # 保存最终数据集（ready_to_train.csv）
    print("\n保存最终数据集...")
    df_final_save = df_final.reset_index()
    df_final_save.to_csv(output_dir / "ready_to_train.csv", index=False)
    print(f"  已保存最终数据集: {output_dir / 'ready_to_train.csv'}")
    
    print(f"\n数据集已保存至: {output_dir}")
    print(f"最终数据集形状: {df_final.shape}")
    print("=" * 60)
    
    return df_final


def load_prepared_data(data_dir: str = "data",
                      filename: str = "ready_to_train.csv") -> pd.DataFrame:
    """
    加载已准备的数据集
    
    参数:
        data_dir: 数据目录
        filename: 文件名（默认 'ready_to_train.csv'）
    
    返回:
        完整的数据集 DataFrame
    """
    data_dir = Path(data_dir)
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.set_index('Date')
    
    print(f"成功加载数据集: {df.shape}")
    print(f"  日期范围: {df.index.min()} 至 {df.index.max()}")
    
    return df


if __name__ == "__main__":
    # 示例用法
    data_path = "project\project\data\lead_futures_historical_data_raw.csv"
    output_path = "project\project\data"
    
    # 准备数据集（30 天后涨跌分类任务）
    df_final = prepare_dataset(
        data_path=data_path,
        output_dir=output_path,
        forward_days=30,  # 预测 30 天后的涨跌
        save_intermediate=True
    )
    
    # 查看数据摘要
    print("\n数据摘要:")
    print(f"  形状: {df_final.shape}")
    print(f"  日期范围: {df_final.index.min()} 至 {df_final.index.max()}")
    # 特征列（排除目标变量）
    feature_cols = [c for c in df_final.columns if c not in ['Target', 'Target_30d_Up', 'Forward_Return_30']]
    print(f"  特征列数: {len(feature_cols)}")
    print(f"  目标变量: Target (30天后涨跌分类)")
    if 'Target' in df_final.columns:
        target_dist = df_final['Target'].value_counts()
        print(f"    涨 (1): {target_dist.get(1, 0)} 条 ({target_dist.get(1, 0)/len(df_final)*100:.1f}%)")
        print(f"    跌 (0): {target_dist.get(0, 0)} 条 ({target_dist.get(0, 0)/len(df_final)*100:.1f}%)")

