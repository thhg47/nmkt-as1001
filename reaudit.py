import pandas as pd
from common import load_data, FEATURES, TARGET

# Định nghĩa chi tiết lý do chuyên nghiệp cho từng trường ngoại lệ
EXCLUDED = {
    'segment_id':     ('yes', 'Unique identifier only; carries no signal-quality information'),
    'participant_id': ('yes', 'Grouping and audit key; model must generalize to unseen participants'),
    'duration_s':     ('yes', 'Constant at 10 for every segment; no information'),
    'review_note':    ('no',  'Written after the quality judgment; perfectly separates the labels in training data'),
    TARGET:           ('no',  'Prediction target; must not be used as an input'),
}

def main():
    # 1. Dùng load_data từ common.py
    df = load_data()
    
    # 2. Sinh tự động bảng từ dữ liệu gốc (không gõ tay tên cột)
    raw_cols = [c for c in df.columns if c != 'split']          # 11 trường gốc
    rows = []
    
    for c in raw_cols:
        if c in EXCLUDED:
            avail, reason = EXCLUDED[c]
            inc = 'no'
        elif c in FEATURES:
            avail, inc, reason = 'yes', 'yes', 'Available at acquisition; segment summary'
        else:
            avail, inc, reason = 'yes', 'no', 'Auxiliary column excluded from modeling'
            
        rows.append({
            'feature': c,
            'available_at_prediction': avail,
            'include': inc,
            'reason': reason
        })
        
    audit = pd.DataFrame(rows)
    audit.to_csv('feature_audit.csv', index=False)

    # 3. Các assertions kiểm tra khắt khe theo đúng yêu cầu kiểm thử
    assert len(audit) == 11, f"Lỗi: Số dòng là {len(audit)}, yêu cầu đúng 11 dòng."
    included_features = set(audit.loc[audit['include'] == 'yes', 'feature'])
    assert included_features == set(FEATURES), "Lỗi: Tập include=yes không khớp với FEATURES trong common.py"
    assert 'split' not in set(audit['feature']), "Lỗi: Cột 'split' không được phép nằm trong audit"
    print(f'feature_audit.csv OK: {len(audit)} rows; {len(FEATURES)} included')

    # 4. Kiểm tra sự tồn tại của cột split trước khi tính toán thống kê nhãn
    assert 'split' in df.columns, 'split missing: refusing to compute label stats without it'
    
    # Chỉ tính toán trên train và validation (tuyệt đối không tính trên test)
    tv = df[df['split'].isin(['train', 'validation'])]
    
    print("\n--- PHÂN BỐ NHÃN TRÊN TRAIN / VALIDATION ---")
    print(tv[TARGET].value_counts())
    
    print("\n--- BẰNG CHỨNG CROSSTAB (REVIEW_NOTE vs QUALITY_LABEL) ---")
    if 'review_note' in df.columns:
        print(pd.crosstab(tv['review_note'], tv[TARGET]))

if __name__ == '__main__':
    main()