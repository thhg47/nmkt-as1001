import pandas as pd
import json
import os

def main():
    data_path = "ecg_quality_features.jsonl"
    split_path = "participant_split.json"
    
    if not os.path.exists(data_path) or not os.path.exists(split_path):
        print("LỖI: Thiếu file dữ liệu hoặc file participant_split.json trong thư mục!")
        return

    # 1. Load dữ liệu và ghép split
    df = pd.read_json(data_path, lines=True)
    with open(split_path, 'r') as f:
        split_data = json.load(f)
        
    mapping = {}
    for split_name, p_list in split_data.items():
        for pid in p_list:
            mapping[pid] = split_name
            
    id_col = next((c for c in df.columns if 'participant' in c.lower() or 'subject' in c.lower() or 'id' in c.lower()), None)
    if id_col:
        df['split'] = df[id_col].map(mapping)
        
    TARGET = "quality_label"
    
    # 2. Tạo file feature_audit.csv đúng chuẩn 4 cột
    audit_data = []
    for col in df.columns:
        available = "yes"
        include = "yes"
        reason = "Valid feature for modeling"
        
        if col in ["review_note", TARGET]:
            available = "no"
            include = "no"
            reason = "Post-hoc label / Target variable, exclude to prevent leakage"
            
        audit_data.append({
            "feature": col,
            "available_at_prediction": available,
            "include": include,
            "reason": reason
        })
        
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv("feature_audit.csv", index=False)
    print(">>> ĐÃ TẠO XONG file 'feature_audit.csv' chuẩn 4 cột!")
    
    # 3. Thống kê nhãn chỉ trên train/validation (bỏ qua test) và crosstab review_note
    if 'split' in df.columns:
        train_val_df = df[df['split'].isin(['train', 'validation', 'val'])]
    else:
        train_val_df = df
        
    print("\n--- PHÂN BỐ NHÃN (TRAIN / VALIDATION) ---")
    if TARGET in train_val_df.columns:
        print(train_val_df[TARGET].value_counts())
    else:
        print(f"Không tìm thấy cột target '{TARGET}'")
        
    if 'review_note' in df.columns and TARGET in df.columns:
        print("\n--- CROSSTAB REVIEW_NOTE vs QUALITY_LABEL ---")
        print(pd.crosstab(train_val_df['review_note'], train_val_df[TARGET]))

if __name__ == "__main__":
    main()