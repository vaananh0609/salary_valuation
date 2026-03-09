import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.db_settings import get_mongo_client, get_raw_collection
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_data_audit():
    """
    Tiến hành kiểm toán chất lượng dữ liệu toàn diện trên vùng Staging
    để xác nhận tính chính xác của quá trình khử trùng lặp.
    """
    logger.info("=" * 80)
    logger.info("BÁO CÁO KIỂM TOÁN DỮ LIỆU (Data Quality Audit)")
    logger.info("=" * 80)

    client = get_mongo_client()
    db = client["jobs"]  # Raw DB name
    raw_col = db["raw_jobs"]
    staging_col = db["staging_jobs"]

    # ============================================
    # 1. KIỂM TRA ĐỊNH LƯỢNG
    # ============================================
    logger.info("\n[1/3] KIỂM TRA ĐỊNH LƯỢNG")
    logger.info("-" * 80)

    raw_count = raw_col.count_documents({})
    staging_count = staging_col.count_documents({})
    dedup_rate = ((raw_count - staging_count) / raw_count * 100) if raw_count > 0 else 0

    logger.info(f"Số bản ghi vùng thô (raw_jobs): {raw_count:,}")
    logger.info(f"Số bản ghi vùng tinh chế (staging_jobs): {staging_count:,}")
    logger.info(f"Số bản ghi bị loại bỏ: {raw_count - staging_count:,}")
    logger.info(f"Tỉ lệ khử trùng: {dedup_rate:.2f}%")

    # ============================================
    # 2. KIỂM TRA TÍNH DUY NHẤT CỦA URL
    # ============================================
    logger.info("\n[2/3] KIỂM TRA TÍNH DUY NHẤT CỦA URL")
    logger.info("-" * 80)

    pipeline_url = [
        {"$group": {"_id": "$link", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
    ]
    duplicate_urls = list(staging_col.aggregate(pipeline_url))

    if duplicate_urls:
        logger.warning(f"[CẢNH BÁO] Phát hiện {len(duplicate_urls)} URL bị trùng lặp vật lý.")
        for i, dup in enumerate(duplicate_urls[:5], 1):
            logger.warning(f"  {i}. {dup['_id']} (xuất hiện {dup['count']} lần)")
    else:
        logger.info("[✓ ĐẠT] Không có URL nào trùng lặp trong tập dữ liệu tiني chế.")

    # ============================================
    # 3. KIỂM TRA CHẤT LƯỢNG NLP & FALSE POSITIVE
    # ============================================
    logger.info("\n[3/3] KIỂM TRA CHẤT LƯỢNG NLP (Phân tích công ty hàng đầu)")
    logger.info("-" * 80)

    pipeline_company = [
        {"$match": {"company": {"$ne": ""}}},
        {
            "$group": {
                "_id": "$company",
                "job_count": {"$sum": 1},
                "titles": {"$push": "$title"},
                "sources": {"$push": "$source"},
            }
        },
        {"$sort": {"job_count": -1}},
        {"$limit": 1},
    ]

    top_company_result = list(staging_col.aggregate(pipeline_company))

    if top_company_result:
        comp = top_company_result[0]
        logger.info(f"\n[PHÂN TÍCH] Công ty có nhiều tin đăng nhất:")
        logger.info(f"  Tên công ty: {comp['_id']}")
        logger.info(f"  Tổng số tin đăng: {comp['job_count']} tin")
        logger.info(f"  Nguồn dữ liệu: {', '.join(set(comp['sources']))}")
        logger.info(f"\n  Danh sách chức danh (5 mẫu ngẫu nhiên để kiểm tra gộp nhầm):")

        # Lấy tối đa 5 chức danh độc lập đầu tiên
        seen_titles = set()
        count = 0
        for title in comp['titles']:
            if title not in seen_titles:
                seen_titles.add(title)
                logger.info(f"    {count + 1}. {title}")
                count += 1
                if count >= 5:
                    break

        logger.info(
            "\n  ✓ Nếu các chức danh trên vẫn giữ được sự đa dạng về kỹ năng (ví dụ: Java Developer,"
            "\n    Python Developer, Frontend, Backend tồn tại độc lập), điều này chứng minh bộ từ"
            "\n    điển tiếng Việt lọc nhiễu hoạt động chuẩn xác và chưa gộp nhầm chuyên môn."
        )

    # ============================================
    # 4. KIỂM TRA CHỈ MỤC (INDEX)
    # ============================================
    logger.info("\n[PHỤ] KIỂM TRA CHỈ MỤC (INDEX)")
    logger.info("-" * 80)

    indices = staging_col.list_indexes()
    for idx in indices:
        logger.info(f"Chỉ mục: {idx['name']} -> {idx['key']}")

    # ============================================
    # KẾT LUẬN
    # ============================================
    logger.info("\n" + "=" * 80)
    logger.info("KẾT LUẬN")
    logger.info("=" * 80)

    if not duplicate_urls and staging_count > 0:
        logger.info("✓ Quá trình khử trùng lặp được đánh giá là AN TOÀN và CHÍNH XÁC.")
        logger.info(
            f"✓ Tập dữ liệu {staging_count:,} bản ghi đã sẵn sàng chuyển giao cho phân hệ"
            " bóc tách từ khóa O*NET ở pha kế tiếp."
        )
    else:
        logger.warning("[!] Phát hiện cảnh báo - cần kiểm tra kỹ trước khi sử dụng dữ liệu.")

    client.close()


if __name__ == "__main__":
    run_data_audit()
