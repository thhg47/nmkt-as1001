# AI Interaction Log (ai_log.md)

This document records the transparent usage of generative AI tools (such as Gemini and ChatGPT) throughout the development of the ECG Quality Assessment project, detailing prompt objectives, inquiries, and human-in-the-loop verification steps.

---

## 1. Ta Nguyen Minh Dang (ID: 2652430)
* **Primary Focus:** Project workflow structuring, repository task distribution, and Git version control operations.
* **Key Prompts & Inquiries:**
  * *“Hãy tóm tắt và đề ra các bước thực hiện bài tập này”(Cung cấp file data)"*
  * *“Phân tích các file cần làm theo nhiệm vụ phân công”(Cung cấp file và list chia việc)"*
  * *“File common và audit (.py) không phải chung 1 code mà phải tách riêng vì sao?”*
  * *“Dữ liệu trong file reaudit.py sao lại khác với file csv?”*
  * *"Làm thế nào để đảm bảo file feature_audit.csv có đúng 11 dòng và chính xác 6 dòng include=yes theo đúng checklist của nhóm test?"*
  * *"Có đúng yêu cầu không?" (Gửi ảnh màn hình console chạy để xác nhận kết quả output và bảng crosstab)."*
  * *"File common.py nó như vậy thì hãy làm cái file kia cho phù hợp..." (Cung cấp mã nguồn chuẩn để khớp code chính xác 100%)."*
  * *"Đúng yêu cầu này chưa..." (Đưa checklist phản hồi khắt khe của team test về 6 đặc trưng và yêu cầu sinh bảng tự động từ dữ liệu chứ không gõ tay)."*
  * *"Làm sao để đồng bộ hóa lại nhánh của mình khi bị tụt lại phía sau (behind) so với nhánh main của nhóm?"*
  * *"Cần dùng những câu lệnh Git nào để xóa các file rác/cũ và cập nhật code mới lên nhánh feature/data-audit?"*
  * *"Các bước chi tiết để bấm tạo Pull Request (PR) trực tiếp trên giao diện web của GitHub là gì để mọi người vào review?"*

---

## 2. Pham Thanh Hang (ID: 2652680)
* **Primary Focus:** Evaluation metric design, validation threshold sweeping, and cost-sensitive optimization.
* **Key Prompts & Inquiries:**
  * *"Hãy hướng dẫn tôi cách trình bày phần Đánh giá mô hình ECG bài bản. Ngoài các chỉ số cơ bản, làm sao để phân tích bổ sung các trường hợp nhiễu sóng ECG khiến mô hình đoán sai?"*
  * *"Tôi cần làm gì để tính toán các hàm như metrics row, boot"*
  * *"Làm sao chạy trên validation?"*
  * *"Check lỗi và đối chiếu dữ liệu với các thành viên khác xem đã khớp chưa
  Chạy thử trên dữ liệu giả lập xem đã hoạt động được chưa"*
  * *"Hướng dẫn và kiểm tra các chỉnh sửa trong threshold.py"*
  * *"check giúp tôi phần threshold và phần training đã đầy đủ theo yêu cầu chưa."*
  * *"Hướng dẫn đối chiếu với phần training model"*
    
---

## 3. Nguyen Duc Minh (ID: 2652458)
* **Primary Focus:** Machine learning pipeline architecture, data leakage prevention, feature preprocessing, and model optimization.
* **Key Prompts & Inquiries:**
  * *"Hãy giải thích nhiệm vụ Baseline và mô hình cuối cần thực hiện những gì? Hướng dẫn các bước xây dựng mô hình và viết phần Method cho báo cáo."*
  * *"Phân tích code hiện tại trong model_training.py, common.py và evaluate.py để xác định những phần cần sửa."*
  * *"Hướng dẫn xây dựng pipeline tiền xử lý dữ liệu bằng StandardScaler, OneHotEncoder và Logistic Regression mà không gây data leakage."*
  * *"Cách sử dụng DummyClassifier làm baseline và thử thêm Random Forest để so sánh?"*
  * *"Chọn siêu tham số C của Logistic Regression bằng tập validation như thế nào?"*
  * *"Cần thực hiện những bước nào để đóng băng (freeze) cấu hình mô hình trước khi đánh giá trên tập test?"*
