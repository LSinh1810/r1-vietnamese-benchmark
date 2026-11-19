# Quy tắc làm việc với Codex

- Tách logic đọc dữ liệu, định dạng prompt, suy luận, chấm điểm và phân tích thành các module Python có trách nhiệm rõ ràng trong `src/` khi triển khai. Ưu tiên tên dễ hiểu, hàm ngắn và chú thích cho các quyết định khó hiểu.
- Dùng notebook để điều phối thực nghiệm, giải thích và trình bày. Mỗi notebook cần có Markdown mô tả mục tiêu, phương pháp và kết quả. Gọi lại logic từ các module Python.
- Nhận đường dẫn từ cấu hình hoặc tham số và dùng đường dẫn tương đối với gốc dự án. Không hardcode đường dẫn cá nhân hoặc thông tin xác thực.
- Chỉ dùng dữ liệu thực nghiệm có nguồn gốc và đáp án đã kiểm tra. Không tạo dữ liệu giả để thay thế dữ liệu thực nghiệm. Tách pilot và test theo họ câu hỏi.
- Giữ nguyên định nghĩa metric, mẫu số và quy tắc chấm đã thống nhất. Mọi thay đổi cần được người dùng chấp thuận và ghi vào giao thức.
- Chốt prompt, giới hạn token, số lần sinh và rubric trên pilot trước khi chạy test. Không dùng kết quả test để chọn cấu hình chính.
- Mỗi lần chạy cần có mã riêng và bản ghi liên kết kết quả với cấu hình đã dùng, ID và revision của model/tokenizer, phiên bản dữ liệu, môi trường, seed và tham số sinh. Không ghi đè kết quả của lần chạy trước.
- Trong giai đoạn thiết lập, chỉ tạo cấu trúc và tài liệu nền tảng. Triển khai inference, evaluation và notebook thực nghiệm khi có yêu cầu tiếp theo.
