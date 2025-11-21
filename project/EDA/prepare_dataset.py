"""
Lead 期货数据准备模块

功能：
1. 数据清洗和预处理
2. 特征工程（技术指标、滞后特征等）
3. 目标变量创建
4. 数据分割（训练/验证/测试）
5. 数据保存和加载
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
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


def create_features(df: pd.DataFrame, 
                   ma_windows: list = [5, 10, 20, 50, 100, 200],
                   return_lags: list = [1, 2, 3, 5, 10],
                   volatility_windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    创建技术指标特征
    
    参数:
        df: 清洗后的 DataFrame
        ma_windows: 移动平均窗口列表
        return_lags: 收益率滞后阶数列表
        volatility_windows: 波动率计算窗口列表
    
    返回:
        包含特征的 DataFrame
    """
    df_features = df.copy()
    
    # 1. 基础收益率
    df_features['Returns'] = df_features['Close'].pct_change()
    df_features['Log_Returns'] = np.log(df_features['Close'] / df_features['Close'].shift(1))
    
    # 2. 移动平均线
    for window in ma_windows:
        df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()
        # 价格相对均线的偏离
        df_features[f'Price_MA_{window}_Ratio'] = df_features['Close'] / df_features[f'MA_{window}'] - 1
    
    # 3. 指数移动平均
    for window in [12, 26]:
        df_features[f'EMA_{window}'] = df_features['Close'].ewm(span=window, adjust=False).mean()
    
    # MACD 指标
    if 'EMA_12' in df_features.columns and 'EMA_26' in df_features.columns:
        df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
        df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
        df_features['MACD_Hist'] = df_features['MACD'] - df_features['MACD_Signal']
    
    # 4. RSI 指标
    delta = df_features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. 布林带
    df_features['BB_Middle'] = df_features['Close'].rolling(window=20).mean()
    bb_std = df_features['Close'].rolling(window=20).std()
    df_features['BB_Upper'] = df_features['BB_Middle'] + 2 * bb_std
    df_features['BB_Lower'] = df_features['BB_Middle'] - 2 * bb_std
    df_features['BB_Width'] = (df_features['BB_Upper'] - df_features['BB_Lower']) / df_features['BB_Middle']
    df_features['BB_Position'] = (df_features['Close'] - df_features['BB_Lower']) / (df_features['BB_Upper'] - df_features['BB_Lower'])
    
    # 6. 波动率
    for window in volatility_windows:
        df_features[f'Volatility_{window}'] = df_features['Returns'].rolling(window=window).std() * np.sqrt(252)
    
    # 7. 价格变化特征
    df_features['High_Low_Ratio'] = df_features['High'] / df_features['Low'] - 1
    df_features['Close_Open_Ratio'] = df_features['Close'] / df_features['Open'] - 1
    df_features['High_Close_Ratio'] = df_features['High'] / df_features['Close'] - 1
    df_features['Low_Close_Ratio'] = df_features['Low'] / df_features['Close'] - 1
    
    # 8. 滚动最高/最低
    for window in [5, 10, 20]:
        df_features[f'High_{window}'] = df_features['High'].rolling(window=window).max()
        df_features[f'Low_{window}'] = df_features['Low'].rolling(window=window).min()
        df_features[f'Close_High_{window}_Ratio'] = df_features['Close'] / df_features[f'High_{window}'] - 1
        df_features[f'Close_Low_{window}_Ratio'] = df_features['Close'] / df_features[f'Low_{window}'] - 1
    
    # 9. 滞后收益率特征
    for lag in return_lags:
        df_features[f'Returns_Lag_{lag}'] = df_features['Returns'].shift(lag)
    
    # 10. 成交量相关特征（如果有成交量数据）
    if 'Volume' in df_features.columns:
        df_features['Volume_MA_20'] = df_features['Volume'].rolling(window=20).mean()
        df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA_20']
        df_features['Price_Volume'] = df_features['Close'] * df_features['Volume']
    
    # 11. 时间特征
    df_features['Year'] = df_features.index.year
    df_features['Month'] = df_features.index.month
    df_features['DayOfWeek'] = df_features.index.dayofweek
    df_features['DayOfMonth'] = df_features.index.day
    
    # 12. 趋势特征
    df_features['Trend_5'] = (df_features['Close'] > df_features['Close'].shift(5)).astype(int)
    df_features['Trend_10'] = (df_features['Close'] > df_features['Close'].shift(10)).astype(int)
    df_features['Trend_20'] = (df_features['Close'] > df_features['Close'].shift(20)).astype(int)
    
    print(f"特征创建完成: {df_features.shape}")
    print(f"特征数量: {len(df_features.columns)}")
    
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


def split_data(df: pd.DataFrame,
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序分割数据
    
    参数:
        df: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    返回:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"数据分割完成:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/n:.1%})")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/n:.1%})")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/n:.1%})")
    print(f"  训练集日期范围: {train_df.index.min()} 至 {train_df.index.max()}")
    print(f"  验证集日期范围: {val_df.index.min()} 至 {val_df.index.max()}")
    print(f"  测试集日期范围: {test_df.index.min()} 至 {test_df.index.max()}")
    
    return train_df, val_df, test_df


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
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   forward_days: int = 30,
                   save_intermediate: bool = True) -> Dict[str, pd.DataFrame]:
    """
    完整的数据准备流程 - 30 天后涨跌分类任务
    
    参数:
        data_path: 原始数据文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        forward_days: 预测未来多少天的涨跌（默认 30 天）
        save_intermediate: 是否保存中间结果
    
    返回:
        包含 train_df, val_df, test_df 的字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始数据准备流程")
    print("=" * 60)
    
    # 1. 加载原始数据
    print("\n[1/6] 加载原始数据...")
    df_raw = load_raw_data(data_path)
    
    # 2. 数据清洗
    print("\n[2/6] 数据清洗...")
    df_clean = clean_data(df_raw)
    if save_intermediate:
        df_clean.to_csv(output_dir / "lead_futures_cleaned.csv")
        print(f"  已保存清洗后的数据: {output_dir / 'lead_futures_cleaned.csv'}")
    
    # 3. 创建特征
    print("\n[3/6] 创建特征...")
    df_features = create_features(df_clean)
    if save_intermediate:
        df_features.to_csv(output_dir / "lead_futures_with_features.csv")
        print(f"  已保存带特征的数据: {output_dir / 'lead_futures_with_features.csv'}")
    
    # 4. 创建目标变量（30 天后涨跌分类）
    print("\n[4/6] 创建目标变量（30 天后涨跌分类）...")
    df_targets = create_targets(df_features, forward_days=forward_days)
    
    # 5. 移除 NaN 行并填充特征缺失值
    print("\n[5/6] 移除 NaN 行并填充特征缺失值...")
    df_final = remove_nan_rows(df_targets, target_column='Target')
    
    # 6. 数据分割
    print("\n[6/6] 数据分割...")
    train_df, val_df, test_df = split_data(df_final, train_ratio, val_ratio, test_ratio)
    
    # 保存最终数据集
    print("\n保存最终数据集...")
    train_df.to_csv(output_dir / "train.csv")
    val_df.to_csv(output_dir / "val.csv")
    test_df.to_csv(output_dir / "test.csv")
    
    # 保存为 pickle 格式（更快，保留数据类型）
    train_df.to_pickle(output_dir / "train.pkl")
    val_df.to_pickle(output_dir / "val.pkl")
    test_df.to_pickle(output_dir / "test.pkl")
    
    print(f"\n数据集已保存至: {output_dir}")
    print("=" * 60)
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'full': df_final
    }


def load_prepared_data(data_dir: str = "data",
                      format: str = "pkl") -> Dict[str, pd.DataFrame]:
    """
    加载已准备的数据集
    
    参数:
        data_dir: 数据目录
        format: 文件格式 ('csv' 或 'pkl')
    
    返回:
        包含 train_df, val_df, test_df 的字典
    """
    data_dir = Path(data_dir)
    
    if format == "pkl":
        train_df = pd.read_pickle(data_dir / "train.pkl")
        val_df = pd.read_pickle(data_dir / "val.pkl")
        test_df = pd.read_pickle(data_dir / "test.pkl")
    else:
        train_df = pd.read_csv(data_dir / "train.csv", index_col=0, parse_dates=True)
        val_df = pd.read_csv(data_dir / "val.csv", index_col=0, parse_dates=True)
        test_df = pd.read_csv(data_dir / "test.csv", index_col=0, parse_dates=True)
    
    print(f"成功加载数据集:")
    print(f"  训练集: {train_df.shape}")
    print(f"  验证集: {val_df.shape}")
    print(f"  测试集: {test_df.shape}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


if __name__ == "__main__":
    # 示例用法
    data_path = "data/lead_futures_historical_data_raw.csv"
    
    # 准备数据集（30 天后涨跌分类任务）
    datasets = prepare_dataset(
        data_path=data_path,
        output_dir="data",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        forward_days=30,  # 预测 30 天后的涨跌
        save_intermediate=True
    )
    
    # 查看数据摘要
    print("\n数据摘要:")
    for name, df in datasets.items():
        if name != 'full':
            print(f"\n{name.upper()} 集:")
            print(f"  形状: {df.shape}")
            print(f"  日期范围: {df.index.min()} 至 {df.index.max()}")
            # 特征列（排除目标变量）
            feature_cols = [c for c in df.columns if c not in ['Target', 'Target_30d_Up', 'Forward_Return_30']]
            print(f"  特征列数: {len(feature_cols)}")
            print(f"  目标变量: Target (30天后涨跌分类)")
            if 'Target' in df.columns:
                target_dist = df['Target'].value_counts()
                print(f"    涨 (1): {target_dist.get(1, 0)} 条 ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
                print(f"    跌 (0): {target_dist.get(0, 0)} 条 ({target_dist.get(0, 0)/len(df)*100:.1f}%)")

