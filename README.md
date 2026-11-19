# Đánh giá DeepSeek-R1-Distill trên bài toán suy luận tiếng Việt

Dự án nghiên cứu zero-shot trên Google Colab, tập trung vào độ chính xác, ngôn ngữ của phần suy luận được sinh ra và câu trả lời cuối, mối liên hệ giữa độ dài suy luận và độ chính xác, cùng các loại lỗi thường gặp. Dự án dùng checkpoint có sẵn, không huấn luyện hoặc fine-tune.

So sánh chính ghép cặp 1.5B và 7B trên cùng 300 câu kiểm thử. 14B là phần mở rộng tùy tài nguyên. Thiết kế dự kiến gồm 40 câu pilot riêng và bốn nhóm test, mỗi nhóm 75 câu gồm toán có lời văn, logic và ràng buộc, suy luận từ đoạn văn, và suy luận định lượng thực tế. Pilot dùng để chốt cấu hình trước khi chạy test.

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| `configs/` | Cấu hình pilot và thực nghiệm. `experiment.yaml` là mẫu cấu hình dự kiến. |
| `data/` | Dữ liệu pilot và test, đáp án chuẩn và thông tin nguồn. |
| `notebooks/` | Notebook Colab điều phối thực nghiệm và trình bày kết quả. |
| `src/` | Module Python dùng lại giữa các notebook. |
| `results/` | Đầu ra và bản ghi cấu hình theo từng mã lần chạy. |
| `reports/` | Báo cáo, bảng và hình phục vụ trình bày nghiên cứu. |

## Môi trường dự kiến

Khi bắt đầu pilot, mở hoặc tải dự án vào Colab, chọn runtime GPU và cài các gói trong `requirements.txt` từ thư mục gốc bằng `python -m pip install -r requirements.txt`. `bitsandbytes` phục vụ phương án lượng tử hóa nếu cần. Ghi lại phiên bản thực tế của thư viện, loại GPU và cấu hình suy luận cho mỗi lần chạy.

## Giao thức và truy xuất kết quả

Chỉ số chính là tỷ lệ trả lời đúng trong một lần sinh. Khi dùng số lần sinh bằng nhau cho mọi câu, tính tỷ lệ đúng theo từng câu rồi lấy trung bình. Báo cáo chênh lệch accuracy ghép cặp giữa các checkpoint và khoảng tin cậy 95%. Đầu ra hợp lệ nhưng thiếu đáp án được tính sai. Sự cố hạ tầng chưa khôi phục được ghi là dữ liệu thiếu và báo cáo riêng.

Mỗi thư mục `results/<run_id>/` sẽ lưu đầu ra thô và bản ghi gồm cấu hình thực tế, phiên bản dữ liệu, model/tokenizer và revision, môi trường, seed, tham số sinh và trạng thái xử lý. Mỗi kết quả trong báo cáo phải truy được về mã lần chạy tương ứng.

Các trường chưa chốt trong cấu hình được để `null` và sẽ được điền khi khóa giao thức sau pilot.

## Khám phá dữ liệu

`notebooks/01_dataset_exploration.ipynb` đọc dữ liệu qua `src/dataset.py`. Đặt `DATASET_PATH` trong notebook để chọn tệp CSV hoặc JSONL UTF-8. Có thể lưu thêm các trường nguồn và họ câu hỏi bên cạnh năm trường bắt buộc `id`, `question`, `answer`, `topic` và `difficulty`.

ID có thể là chuỗi hoặc số nguyên. Đáp án có thể là chuỗi, số nguyên hoặc số thực hữu hạn. Câu hỏi và nhãn phải là chuỗi. Nhãn độ khó là `dễ`, `trung bình` hoặc `khó`. Module giữ thứ tự bản ghi và báo cáo lỗi để người nghiên cứu xem xét. Phân phối chủ đề và độ khó bao gồm cả bản ghi có lỗi để làm rõ tình trạng dữ liệu đầu vào.

Chế độ mặc định dùng `data/fixtures/SYNTHETIC_pipeline_check.csv`, chứa dữ liệu giả và lỗi cố ý để kiểm thử. Chỉ dùng bộ câu hỏi đã được kiểm chứng cho pilot và test.

## Inference trên Colab

Ba checkpoint dùng `Qwen2ForCausalLM`, được hỗ trợ trong [Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen2). `requirements.txt` ghim phiên bản Transformers này. Nếu cần giảm VRAM, thử `quantization: '4bit'` với NF4 trong pilot. Khả năng chạy phụ thuộc GPU, CUDA, độ dài đầu ra và batch size. [Tài liệu bitsandbytes cho Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes)

Điền `dataset.path`, `inference.max_new_tokens` và `inference.seeds` trong YAML. Số seed phải bằng `generations_per_question`. `dtype: null` dùng `float16`. Giữ cấu hình thực tế và revision cố định khi chuyển sang test. Notebook `notebooks/02_model_inference.ipynb` hướng dẫn chạy trên GPU. Có thể chạy trực tiếp từ gốc dự án:

```sh
python -m src.inference --config configs/experiment.yaml --model-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --run-id pilot-1p5b-01
```

Kết quả nằm trong `results/<run_id>/`, với manifest, cấu hình sinh hiệu lực, checkpoint JSONL theo batch và `predictions.jsonl` sau khi hoàn tất. Trên Colab, dùng `--output-dir` hoặc biến `OUTPUT_DIR` trong notebook để lưu trên ổ đĩa bền vững. Chạy lại cùng model, dataset, cấu hình và mã lần chạy để tiếp tục. Dùng mã mới khi đổi cấu hình hoặc cần chạy lại từ đầu. Khi một phiên hết VRAM, các batch đã ghi vẫn có thể dùng để tiếp tục trên GPU đủ bộ nhớ.

## Đánh giá kết quả

Mở `notebooks/03_evaluation.ipynb` và đặt `RUN_DIR`, `DATASET_PATH` và, nếu đã chấm tay, `REVIEW_PATH`. Notebook xác minh dataset bằng manifest, hiển thị accuracy, độ dài, ngôn ngữ, tỷ lệ lỗi và danh sách đầu ra cần rà soát. Accuracy chính chỉ xuất hiện khi đủ lượt dự kiến và đã xử lý hết các ca chờ chấm.

Các trường dataset tùy chọn `accepted_answers` (mảng JSON, kể cả khi lưu trong ô CSV) và `absolute_tolerance` (số không âm) ghi các ngoại lệ đã duyệt trong rubric. Mặc định dùng so sánh số chính xác. Nhãn lỗi nội dung được gắn thủ công, còn nhận diện tiếng Việt, tiếng Anh và trộn ngôn ngữ dùng heuristic được mô tả trong notebook.

Chạy các unit test từ gốc dự án bằng `python -m unittest discover -s tests -v`.

## Trực quan hóa kết quả đã lưu

Chạy cell cuối notebook 03 để lưu `evaluation.json` vào thư mục lần chạy. Mở `notebooks/04_visualization.ipynb` và thêm các thư mục đã có inference và evaluation vào `RUN_DIRS`. Chọn một lần chạy mỗi model trên cùng dataset và cùng số lần sinh. Nếu cập nhật điểm sau khi chấm tay hoặc inference có thêm lượt, xuất tệp evaluation với tên mới và chọn tệp đó trong `EVALUATION_FILES`.
