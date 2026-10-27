import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình giao diện biểu đồ
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

def run_data_audit(file_path):
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file '{file_path}'!")
        return
    
    print(f"Đang tải dữ liệu từ: {file_path}...")
    # Dùng read_json với lines=True vì đây là file .jsonl
    df = pd.read_json(file_path, lines=True)
    print(f"-> Tải thành công! Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.\n")
    
    # Kiểm tra giá trị thiếu
    print("="*50)
    print("KIỂM TRA GIÁ TRỊ THIẾU (MISSING VALUES):")
    print("="*50)
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Percentage': (missing / len(df)) * 100})
    print(missing_df[missing_df['Missing_Count'] > 0] if missing.sum() > 0 else "Không có giá trị thiếu nào.")
    
    # Tạo và xuất file feature_audit.csv
    audit_list = []
    for col in df.columns:
        audit_list.append({
            'feature_name': col,
            'data_type': str(df[col].dtype),
            'missing_count': int(df[col].isnull().sum()),
            'missing_percentage': round(float(df[col].isnull().mean() * 100), 2),
            'unique_values': int(df[col].nunique())
        })
        
    feature_audit_df = pd.DataFrame(audit_list)
    feature_audit_df.to_csv('feature_audit.csv', index=False)
    print("\n[HOÀN TẤT] Đã tạo và lưu file 'feature_audit.csv' thành công!")
    
    # Vẽ biểu đồ và lưu ảnh vào thư mục plots
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(14, 6))
    
    target_col = next((c for c in df.columns if c.lower() in ['label', 'quality', 'target', 'is_noisy']), None)
    if target_col:
        plt.subplot(1, 2, 1)
        sns.countplot(data=df, x=target_col, palette='Set2')
        plt.title(f'Phân bố nhãn ({target_col})')
        plt.xlabel('Nhãn')
        plt.ylabel('Số lượng mẫu')
    
    ratio_col = next((c for c in df.columns if 'ratio' in c.lower() or 'variation' in c.lower()), None)
    if ratio_col:
        plt.subplot(1, 2, 2)
        sns.histplot(df[ratio_col].dropna(), kde=True, color='purple', bins=30)
        plt.title(f'Phân phối đặc trưng: {ratio_col}')
        plt.xlabel(ratio_col)
        plt.ylabel('Tần suất')
    
    plt.tight_layout()
    plt.savefig('plots/data_audit_overview.png', dpi=300)
    plt.show()
    print("[HOÀN TẤT] Đã lưu biểu đồ vào thư mục 'plots/data_audit_overview.png'!")

if __name__ == "__main__":
    run_data_audit('ecg_quality_features.jsonl')